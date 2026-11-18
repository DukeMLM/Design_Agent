# tests/test_fixes.py
"""
Small offline tests for the agent system.

No OpenAI key and no AIDE search are needed: the AIDE search is replaced by a
simple training script (a small MLP) and the LLM calls by scripted answers.
Everything else is the real code: dataset generation, the code-data loop, the
script runner, the code verification and the Neural-Adjoint inverse design.

Run from the agent folder:

    python tests/test_fixes.py

The tests use a 2,000-sample subset of the real ADM split in ./dataset when it
exists (see prepare_dataset.py), otherwise a synthetic dataset of the same shape.
All files are written to a temporary folder.
"""

import asyncio
import contextlib
import glob
import inspect
import json
import os
import shutil
import sys
import tempfile
import traceback

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_KEY = "sk-test-not-a-real-key"
os.environ["OPENAI_API_KEY"] = TEST_KEY
os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"
os.environ.setdefault("RUN_TIMEOUT_S", "900")
sys.path.insert(0, AGENT_DIR)

import numpy as np

N_TRAIN, N_VAL = 2000, 400
DIM_G, DIM_S = 14, 2001

# ----------------------------------------------------------------------------
# Stand-in for the AIDE search: the kind of flat script AIDE writes.
# ----------------------------------------------------------------------------
_TRAINING_BODY = '''
torch.manual_seed(0)
np.random.seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
g_train = pd.read_csv("./input/g_training.csv").values.astype("float32")
s_train = pd.read_csv("./input/s_training.csv").values.astype("float32")
g_val = pd.read_csv("./input/g_validation.csv").values.astype("float32")
s_val = pd.read_csv("./input/s_validation.csv").values.astype("float32")
x_scaler = StandardScaler().fit(g_train)
y_scaler = StandardScaler().fit(s_train)
X_tr = torch.tensor(x_scaler.transform(g_train), dtype=torch.float32, device=device)
Y_tr = torch.tensor(y_scaler.transform(s_train), dtype=torch.float32, device=device)
X_va = torch.tensor(x_scaler.transform(g_val), dtype=torch.float32, device=device)
model = MODEL_CLASS(g_train.shape[1], s_train.shape[1]).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(X_tr.shape[0], device=device)
    for i in range(0, X_tr.shape[0], 128):
        idx = perm[i:i + 128]
        if idx.numel() < 2:
            continue
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(X_tr[idx]), Y_tr[idx])
        loss.backward()
        opt.step()
model.eval()
with torch.no_grad():
    pred = y_scaler.inverse_transform(model(X_va).cpu().numpy())
val_mse = float(np.mean((pred - s_val) ** 2))
'''

_MODEL_DEF = '''
class MODEL_CLASS(nn.Module):
    def __init__(self, d_in=14, d_out=2001, width=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, width), nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, d_out),
        )

    def forward(self, x):
        return self.net(x)
'''

_IMPORTS = '''import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import StandardScaler
'''


def simple_aide_script(epochs: int = 15) -> str:
    """A flat training script: no main guard, its own class name, writes ./working/submission.csv."""
    code = _IMPORTS + _MODEL_DEF + _TRAINING_BODY + '''
print(f"Validation MSE: {val_mse:.6f}")
pd.DataFrame(pred).to_csv("./working/submission.csv", index=False)
'''
    return code.replace("MODEL_CLASS", "ResNetLike").replace("EPOCHS", str(epochs))


def refactored_script(epochs: int = 150) -> str:
    """What the CodeModifier has to return for the script above."""
    body = "\n".join("    " + line if line else line for line in _TRAINING_BODY.splitlines())
    code = _IMPORTS + _MODEL_DEF + '\nif __name__ == "__main__":' + body + '''
    torch.save(model.state_dict(), "./model_final.pth")
    joblib.dump(x_scaler, "./x_scaler.save")
    joblib.dump(y_scaler, "./y_scaler.save")
    print(f"Final Validation MSE: {val_mse}")
'''
    return code.replace("MODEL_CLASS", "forward_model").replace("EPOCHS", str(epochs))


# ----------------------------------------------------------------------------
# Test data
# ----------------------------------------------------------------------------
def _copy_head(src: str, dst: str, rows: int) -> None:
    with open(src, "r") as fin, open(dst, "w") as fout:
        for i, line in enumerate(fin):
            if i >= rows:
                break
            fout.write(line)


def _synthetic(n: int, rng) -> tuple:
    """Smooth geometry -> spectrum map with the shape of the ADM data (14 -> 2001)."""
    lo = np.array([0.3, 1.0] + [0.05] * 8 + [-0.8] * 4, dtype="float32")
    hi = np.array([0.6, 1.5] + [0.25] * 8 + [0.8] * 4, dtype="float32")
    g = rng.uniform(lo, hi, size=(n, DIM_G)).astype("float32")
    u = (g - lo) / (hi - lo)
    f = np.linspace(0.0, 1.0, DIM_S, dtype="float32")[None, :]
    s = np.zeros((n, DIM_S), dtype="float32")
    for k in range(4):
        center = 0.15 + 0.7 * (0.5 * u[:, k:k + 1] + 0.5 * u[:, 10 + k:11 + k])
        width = 0.02 + 0.06 * u[:, 4 + k:5 + k]
        height = 0.2 + 0.6 * u[:, 6 + k:7 + k]
        s += height * width ** 2 / ((f - center) ** 2 + width ** 2)
    return g, np.clip(s, 0.0, 1.0)


def make_workdir() -> tuple:
    work = tempfile.mkdtemp(prefix="design_agent_test_")
    os.makedirs(os.path.join(work, "dataset"))
    src = os.path.join(AGENT_DIR, "dataset")
    names = ["g_training.csv", "s_training.csv", "g_validation.csv", "s_validation.csv"]
    rows = [N_TRAIN, N_TRAIN, N_VAL, N_VAL]
    target = os.path.join(work, "dataset", "target_s.csv")
    if all(os.path.exists(os.path.join(src, n)) for n in names):
        for name, n in zip(names, rows):
            _copy_head(os.path.join(src, name), os.path.join(work, "dataset", name), n)
        shutil.copy(os.path.join(src, "test_s_inverse_test_0.csv"), target)
        source = "real ADM data (subset)"
    else:
        rng = np.random.default_rng(0)
        g, s = _synthetic(N_TRAIN + N_VAL + 1, rng)
        parts = [g[:N_TRAIN], s[:N_TRAIN], g[N_TRAIN:N_TRAIN + N_VAL], s[N_TRAIN:N_TRAIN + N_VAL]]
        for name, array in zip(names, parts):
            np.savetxt(os.path.join(work, "dataset", name), array, delimiter=",", fmt="%.8e")
        np.savetxt(target, s[-1:], delimiter=",", fmt="%.8e")
        source = "synthetic data (run prepare_dataset.py to test on the ADM data)"
    return work, source, target


# ----------------------------------------------------------------------------
# Scripted replacements for the LLM
# ----------------------------------------------------------------------------
class FakeResult:
    def __init__(self, text):
        self.final_output = text


class FakeRunner:
    """Replaces agents.Runner inside one module: returns scripted answers and records the prompts."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []
        self.inputs = []

    async def run(self, agent, user_input, **kwargs):
        self.prompts.append(agent.instructions)
        self.inputs.append(user_input)
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        return FakeResult(answer)


TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def rows(path):
    from common import read_table
    return read_table(path).shape[0]


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
@test
def api_key_is_not_overwritten():
    """Importing every module leaves OPENAI_API_KEY as it was."""
    import config, common, task_desp, subagent, forward_loop, modifier, inverse_NA, agent_system  # noqa: F401
    assert os.environ["OPENAI_API_KEY"] == TEST_KEY, "a module overwrote OPENAI_API_KEY"
    for path in glob.glob(os.path.join(AGENT_DIR, "*.py")):
        with open(path, encoding="utf-8") as f:
            source = f.read()
        assert "openai.api_key" not in source and "OPENAI_API_KEY'] =" not in source, path


@test
def dataset_files_with_and_without_header():
    """The raw files (no header) and the ./input files (pandas header) are both read in full."""
    import forward_loop
    from common import has_header, read_table

    assert not has_header("./dataset/g_training.csv")
    assert rows("./dataset/g_training.csv") == N_TRAIN, "the first sample of the raw file was dropped"
    written = forward_loop.generate_dataset(300)
    assert written == 300
    assert has_header("./input/g_training.csv")
    assert read_table("./input/g_training.csv").shape == (300, DIM_G)
    assert read_table("./input/s_training.csv").shape == (300, DIM_S)
    assert read_table("./input/g_validation.csv").shape == (30, DIM_G)
    # a request larger than the dataset is capped and reported
    assert forward_loop.generate_dataset(10 ** 9) == N_TRAIN
    assert rows("./input/g_training.csv") == N_TRAIN


@test
def script_runner():
    """Exit code, time limit, ./working and temp-file cleanup of run_script."""
    from common import run_script

    before = set(glob.glob(os.path.join(tempfile.gettempdir(), "*.py")))
    ok = run_script("open('./working/x.txt', 'w').write('1'); print('héllo')")
    assert ok["ok"] and "llo" in ok["stdout"], ok
    assert os.path.exists("./working/x.txt")
    bad = run_script("print('Epoch 3 Val MSE: 0.5'); raise RuntimeError('boom')")
    assert not bad["ok"] and bad["returncode"] != 0 and "boom" in bad["stderr"]
    slow = run_script("import time; time.sleep(30)", timeout=2)
    assert not slow["ok"] and "TimeoutError" in slow["stderr"]
    assert set(glob.glob(os.path.join(tempfile.gettempdir(), "*.py"))) == before, "temp script left behind"


@test
def mse_parsing_returns_a_float():
    import forward_loop

    value = asyncio.run(forward_loop.parse_mse_async("epoch 9\nValidation MSE: 0.004200\n"))
    assert isinstance(value, float) and abs(value - 0.0042) < 1e-12


@test
def director_prompt_has_the_real_numbers():
    """The threshold and the size limit reach the LoopDirector prompt; the user's target is used."""
    import config, forward_loop

    fake = FakeRunner(['{"dataset_size": 800, "action": "test", "reason": "r"}'])
    original = forward_loop.Runner
    forward_loop.Runner = fake
    try:
        decision = asyncio.run(forward_loop.decide_next_step([{"size": 500, "mse": 0.01}], 0.00321, 12345))
    finally:
        forward_loop.Runner = original
    prompt = fake.prompts[0]
    assert decision["action"] == "test"
    assert "<<" not in prompt and ">>" not in prompt, "unfilled placeholder in the prompt"
    assert prompt.count("0.00321") == 2 and prompt.count("12345") == 2, "threshold / max size missing"

    if os.path.exists(config.TASK_CONFIG_FILE):
        os.remove(config.TASK_CONFIG_FILE)
    assert forward_loop.load_target_mse() == config.MSE_TARGET
    with open(config.TASK_CONFIG_FILE, "w") as f:
        json.dump({"target_mse": 0.004}, f)
    assert forward_loop.load_target_mse() == 0.004


def _run_loop(solutions, decisions, initial_size, maxsize):
    """Run the real code_data_loop with scripted AIDE solutions and LoopDirector decisions."""
    import config, forward_loop

    solutions, decisions = list(solutions), list(decisions)
    seen = {"args": [], "best_at_decision": []}

    def fake_generate(size):
        return dict(solutions.pop(0))

    async def fake_decide(history, mse_threshold, max_size):
        seen["args"].append((mse_threshold, max_size))
        with open(config.BEST_CODE_FILE, encoding="utf-8") as f:
            seen["best_at_decision"].append(f.read())
        return dict(decisions.pop(0))

    originals = forward_loop.generate_solution, forward_loop.decide_next_step
    forward_loop.generate_solution, forward_loop.decide_next_step = fake_generate, fake_decide
    try:
        result = asyncio.run(forward_loop.code_data_loop(initial_size, maxsize=maxsize))
    finally:
        forward_loop.generate_solution, forward_loop.decide_next_step = originals
    return result, seen


@test
def loop_stops_cleanly_when_first_decision_is_done():
    """Was: UnboundLocalError on next_size."""
    import forward_loop

    forward_loop.generate_dataset(300)
    result, seen = _run_loop(
        [{"mse": 0.0015, "code": "CODE_A"}], [{"action": "done", "reason": "target met"}], 300, 1000
    )
    assert result["size"] == 300 and result["mse"] == 0.0015 and result["code"] == "CODE_A"
    assert seen["args"][0] == (0.004, 1000), "target from task_config.json / maxsize not passed on"


@test
def loop_keeps_the_better_code():
    """Was: newly generated code always replaced the code in use."""
    import config

    result, seen = _run_loop(
        [{"mse": 0.010, "code": "CODE_A"}, {"mse": 0.020, "code": "CODE_B"}, {"mse": 0.001, "code": "CODE_C"}],
        [
            {"action": "generate", "dataset_size": 400, "reason": "try again"},
            {"action": "generate", "dataset_size": 400, "reason": "try again"},
            {"action": "done"},
        ],
        300, 1000,
    )
    history = result["history"]
    assert [h.get("kept_code") for h in history] == [None, "previous", "new"], history
    assert seen["best_at_decision"] == ["CODE_A", "CODE_A", "CODE_C"], "best_solution.py not kept current"
    assert result["code"] == "CODE_C" and result["mse"] == 0.001
    with open(config.HISTORY_FILE) as f:
        assert json.load(f) == history, "run_history.json is not the full history"


@test
def loop_tests_real_script_on_growing_data():
    """Real training runs: float MSEs, size capped at the maximum, loop ends at the maximum."""
    script = simple_aide_script(epochs=15)
    result, _ = _run_loop(
        [{"mse": 0.05, "code": script}],
        [
            {"action": "test", "dataset_size": 400, "reason": "more data"},
            {"action": "test", "dataset_size": 10 ** 9, "reason": "far too large"},
            {"action": "done"},
        ],
        300, 600,
    )
    history = result["history"]
    assert [h["size"] for h in history] == [300, 400, 600], history
    assert [h["action"] for h in history] == ["generated", "tested", "tested"], history
    assert all(isinstance(h["mse"], float) for h in history), "MSE stored as text"
    assert result["size"] == 600 and rows("./input/g_training.csv") == 600
    assert result["mse"] == history[-1]["mse"]


@test
def loop_survives_a_crashing_script():
    """Was: the MSE of a crashed run was taken from a line printed before the crash."""
    crashing = "print('Epoch 3 Val MSE: 0.0191')\nraise RuntimeError('out of memory')\n"
    result, _ = _run_loop(
        [{"mse": 0.05, "code": crashing}],
        [{"action": "test", "dataset_size": 400, "reason": "more data"}, {"action": "done"}],
        300, 1000,
    )
    last = result["history"][-1]
    assert last["action"] == "test_failed" and last["mse"] is None, last
    assert "out of memory" in last["reason"]
    assert result["mse"] == 0.05, "the MSE printed before the crash was accepted"


@test
def aide_uses_a_single_validation_split():
    """AIDE_K_FOLD=1 removes the k-fold instruction from AIDE's prompt (no LLM call is made)."""
    try:
        import aide  # noqa: F401
    except Exception as e:  # AIDE is not installed here
        return f"SKIPPED: aide cannot be imported ({type(e).__name__}: {e})"
    import config, forward_loop

    exp = forward_loop._new_experiment("Predict the spectrum from the geometry.")
    try:
        guideline = json.dumps(exp.agent._prompt_impl_guideline)
        task_desc = str(exp.task_desc)
        assert exp.agent.acfg.k_fold_validation == 1
        assert "cross-validation" not in guideline, "AIDE still asks for k-fold cross-validation"
        assert "validation set" in task_desc and "Validation MSE: <value>" in task_desc
        assert exp.interpreter.timeout == config.AIDE_EXEC_TIMEOUT
    finally:
        exp.interpreter.cleanup_session()


@test
def inverse_design_reports_a_missing_model():
    import inverse_NA

    assert not os.path.exists("./modified.py")
    try:
        inverse_NA.load_forward_model_class()
    except FileNotFoundError as e:
        assert "modified.py" in str(e)
    else:
        raise AssertionError("no error for a missing modified.py")


@test
def code_verification_rejects_and_accepts():
    """Structure check before any run, fresh artifacts required, failure is reported as a failure."""
    import config, forward_loop, modifier, subagent

    forward_loop.generate_dataset(N_TRAIN)
    forward_loop.save_code(simple_aide_script())
    assert subagent.missing_files("inverse") == [], subagent.missing_files("inverse")

    runs = {"n": 0}
    original_run = modifier._run_code

    def counting_run(code):
        runs["n"] += 1
        return original_run(code)

    original_runner = modifier.Runner
    modifier._run_code = counting_run
    try:
        # 1) every answer runs fine but saves nothing -> not accepted, nothing written
        useless = 'class forward_model:\n    pass\n\nif __name__ == "__main__":\n    print("Final Validation MSE: 0.1")\n'
        modifier.Runner = FakeRunner([useless])
        try:
            asyncio.run(subagent.run_code_verify())
        except RuntimeError as e:
            assert "could not be verified" in str(e) and "model_final.pth" in str(e), e
        else:
            raise AssertionError("a script that saved no model was accepted")
        assert not os.path.exists(config.MODIFIED_CODE_FILE), "modified.py written after a failed verification"
        assert runs["n"] == 4

        # 2) fenced script without a main guard is rejected without a training run, then a good one passes
        runs["n"] = 0
        unguarded = simple_aide_script().replace("ResNetLike", "forward_model")
        fake = FakeRunner(["```python\n" + unguarded + "\n```", refactored_script()])
        modifier.Runner = fake
        message = asyncio.run(subagent.run_code_verify())
        assert "modified.py" in message
        assert runs["n"] == 1, f"expected one training run, got {runs['n']}"
        assert "__main__" in fake.inputs[1], "the missing main guard was not reported back"
        for path in (config.MODIFIED_CODE_FILE, config.MODEL_FILE, config.X_SCALER_FILE, config.Y_SCALER_FILE):
            assert os.path.exists(path), path

        # 3) a second call does not retrain
        assert "already verified" in asyncio.run(subagent.run_code_verify())
        assert runs["n"] == 1
    finally:
        modifier._run_code = original_run
        modifier.Runner = original_runner


@test
def inverse_design_uses_the_trained_model():
    """Trained weights, eval mode, design inside the training range, plot in physical units."""
    import joblib
    import torch
    import config
    import inverse_NA
    from common import read_table

    target_path = "./dataset/target_s.csv"
    x_scaler = joblib.load(config.X_SCALER_FILE)
    y_scaler = joblib.load(config.Y_SCALER_FILE)
    s_raw = read_table(target_path)
    s_scaled = np.asarray(y_scaler.transform(s_raw), dtype="float32")
    lower, upper = inverse_NA.geometry_bounds(x_scaler)
    forward_model = inverse_NA.load_forward_model_class()

    def evaluator():
        return inverse_NA.NeuralAdjointEvaluator(
            forward_model().to(inverse_NA.DEVICE), forward_model().to(inverse_NA.DEVICE), None,
            config.MODEL_FILE, inverse_NA._model_config(lower, upper), x_scaler, y_scaler,
        )

    # the fixed path: evaluate_one loads the weights itself
    torch.manual_seed(0)
    ev = evaluator()
    xpred, ypred, mse = ev.evaluate_one(s_scaled)
    saved = torch.load(config.MODEL_FILE, map_location="cpu")
    for net in (ev.model, ev.simulator):
        assert not net.training, "model left in training mode"
        assert not any(p.requires_grad for p in net.parameters())
        state = net.state_dict()
        assert all(torch.equal(state[k].cpu(), saved[k]) for k in saved), "trained weights not loaded"

    # the old behaviour: fresh random weights, training mode
    torch.manual_seed(0)
    old = evaluator()
    old._loaded = True
    _, _, mse_untrained = old.evaluate_one(s_scaled)
    assert mse < mse_untrained, (mse, mse_untrained)

    # end to end, recording what is plotted
    plotted = {}
    original_plot = inverse_NA.plt.plot

    def recording_plot(y, *args, **kwargs):
        plotted[kwargs.get("label")] = np.asarray(y)
        return original_plot(y, *args, **kwargs)

    inverse_NA.plt.plot = recording_plot
    try:
        result = inverse_NA.inverse_one(target_path)
    finally:
        inverse_NA.plt.plot = original_plot

    assert set(result) == {"Surrogate MSE", "geometry"}, result
    assert isinstance(result["Surrogate MSE"], float) and np.isfinite(result["Surrogate MSE"])
    geometry = np.array(result["geometry"])
    g_train = read_table("./input/g_training.csv")
    tol = 1e-4 * (g_train.max(axis=0) - g_train.min(axis=0)) + 1e-6
    assert np.all(geometry >= g_train.min(axis=0) - tol) and np.all(geometry <= g_train.max(axis=0) + tol), \
        "designed geometry outside the training range"
    assert np.allclose(plotted["Real Values"], s_raw[0]), "target plotted in scaled units"
    assert plotted["Prediction"].shape == (DIM_S,)
    assert os.path.exists("./real_inverse.png")
    return (f"surrogate MSE {result['Surrogate MSE']:.5f} with the trained model "
            f"vs {mse_untrained:.5f} with untrained weights (the old behaviour)")


@test
def tools_report_a_failure_once():
    """Long-running tools answer FAILED instead of the SDK's "Please try again"."""
    from agents import RunContextWrapper
    import forward_loop, subagent

    async def boom(*args, **kwargs):
        raise RuntimeError("AIDE crashed")

    original = forward_loop.code_data_loop
    forward_loop.code_data_loop = boom
    try:
        out = asyncio.run(subagent.code_data_loops.on_invoke_tool(RunContextWrapper(context=None), '{"dataset_size": 300}'))
    finally:
        forward_loop.code_data_loop = original
    assert out.startswith("FAILED") and "AIDE crashed" in out and "try again" not in out.lower(), out

    out = asyncio.run(subagent.check_stage_files.on_invoke_tool(RunContextWrapper(context=None), '{"purpose": "inverse"}'))
    assert "True" in str(out) or '"ok": true' in str(out).lower(), out


@test
def entry_points_and_prompts():
    """The __main__ blocks run coroutines; the guard raises; the prompts match the tools."""
    import agent_system, modifier, subagent, task_desp

    for fn in (modifier.modify_and_verify, subagent.run_code_verify, task_desp.run_spec_writer, agent_system.main):
        assert inspect.iscoroutinefunction(fn), fn

    async def call_guard():
        with open(os.devnull, "w", encoding="utf-8") as devnull, \
                contextlib.redirect_stderr(devnull), contextlib.redirect_stdout(devnull):
            agent_system._guard_run_sync()  # prints the call stack before raising

    try:
        asyncio.run(call_guard())
    except RuntimeError as e:
        assert "Nested event loop" in str(e)
    else:
        raise AssertionError("the run_sync guard did not raise under a running loop")

    planner = agent_system.planner_agent.instructions
    assert "code_aideml" not in planner and "generate_desp()" in planner
    assert [t.name for t in agent_system.planner_agent.tools] == [
        "ask_human", "generate_desp", "verify_paths", "run_forward", "run_inverse"]
    inverse = subagent.InverseDesigner.instructions
    assert "Real MSE" not in inverse and "inverse_performance_mse" not in inverse
    assert "from modified import forward_model" in modifier._SYSTEM_PROMPT
    assert "2001D" not in task_desp.TASK_AGENT.instructions


# ----------------------------------------------------------------------------
def main() -> int:
    work, source, _ = make_workdir()
    sys.path.insert(0, work)
    os.chdir(work)
    print(f"Test folder: {work}\nData: {source}\n")
    failed = 0
    for fn in TESTS:
        try:
            note = fn()
            status = "SKIP" if isinstance(note, str) and note.startswith("SKIPPED") else "PASS"
            print(f"[{status}] {fn.__name__}" + (f"  ({note})" if note else ""))
        except Exception:
            failed += 1
            print(f"[FAIL] {fn.__name__}\n{traceback.format_exc()}")
    os.chdir(AGENT_DIR)
    shutil.rmtree(work, ignore_errors=True)
    print(f"\n{len(TESTS) - failed} of {len(TESTS)} passed" + (f", {failed} FAILED" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
