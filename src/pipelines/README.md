
A `Pipeline` can be constructed around any speech model defined as a `torch.nn.Module` by wrapping the model with a task-specific `Model` subclass:

```python
from src.pipelines import Pipeline
from src.models import SpeakerVerificationModel
from src.models.speaker import ResNetSE34V2

model = ResNetSE34V2()

wrapped_model = SpeakerVerificationModel(
    model
)

pipeline = Pipeline(
    model=wrapped_model
)
```

A `Pipeline` object may also hold any of the following components:
* A `Simulation` object defining an acoustic simulation
* A `Preprocessor` object consisting of one or more preprocessing stages
* A `Defense` object consisting of one or more adversarial defenses

For additional documentation on the `Pipeline` class, see here.



```python
# Load simulation
simulation = load_simulation(config)

# Load preprocessing
preprocessor = load_preprocess(config)

# Load model
model = load_model(config)
assert isinstance(model, SpeakerVerificationModel)

# Load adversarial defenses
defense = load_defense(config)
```


See additional documentation on <a href="src/models/README.md">models</a>, <a href="src/simulation/README.md">acoustic simulation</a>, <a href="src/preprocess/README.md">preprocessing</a>, and <a href="src/defenses/README.md">defenses</a>.


For a quick start, load data and pipelines from a ready-made configuration:

```python

# CODE EXAMPLE OF PIPELINE/DATA BUILDER from config file

```




<h3 id="usage-pipelines">Building A Pipeline</h3>

We provide a simple interface for performing adversarial attacks on speech systems. Acoustic simulation, preprocessing, purification-based defenses, detection-based defenses, and models are implemented as differentiable modules and wrapped within a single `Pipeline` object. A `Pipeline` can be constructed around any speech model defined as a `torch.nn.Module` using a task-specific wrapper:

```python
from src.pipelines import Pipeline
from src.models import SpeakerVerificationModel
from src.models.speaker import ResNetSE34V2

model = ResNetSE34V2()

wrapped_model = SpeakerVerificationModel(
    model
)

pipeline = Pipeline(
    model=wrapped_model
)
```

A `Pipeline` object may also hold any of the following components:
* A `Simulation` object defining an acoustic simulation
* A `Preprocessor` object consisting of one or more preprocessing stages
* A `Defense` object consisting of one or more adversarial defenses

```python
from src.preprocess import Preprocessor, Normalize, KaldiStyleVAD
from src.simulation import Simulation, Offset, Bandpass

simulation = Simulation(
    Offset(length=[-.15, .15]),
    Bandpass(low=200, high=6000)
)

preprocessor = Preprocessor(
    Normalize(),
    KaldiStyleVAD()
)

pipeline = Pipeline(
    model=wrapped_model,
    simulation=simulation,
    preprocessor=preprocessor
)
```
