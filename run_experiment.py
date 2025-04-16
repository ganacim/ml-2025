import concurrent.futures
import subprocess


# TODO: Dropout or not, Batchnorm or not, DataAugmentation
def create_command(
    EPOCHS: int,
    SEED: int = None,
    device: str = "cuda",
    LEARNING_RATE: float = None,
    BATCH_SIZE: int = None,
    CHECK_POINT: str = None,
):
    command = "mlc model.train"

    # Add required parameter
    command += f" --epoch {EPOCHS}"

    # Add optional parameters if they are provided
    if SEED is not None:
        command += f" --seed {SEED}"

    if device is not None:
        command += f" --device {device}"

    if LEARNING_RATE is not None:
        command += f" --lr {LEARNING_RATE}"

    if BATCH_SIZE is not None:
        command += f" --batch-size {BATCH_SIZE}"

    if CHECK_POINT is not None:
        command += f" --checkpoint {CHECK_POINT}"

    # Add model and dataset names
    command += " -t VGG cats_and_dogs"

    return command


commands = []


# batch_size experiment

epochs = 10
for batch_size in [16, 32, 64, 128, 256]:
    for lr in [0.0001, 0.001, 0.01, 0.1]:
        command = create_command(
            EPOCHS=epochs,
            device="cuda",
            LEARNING_RATE=lr,
            BATCH_SIZE=batch_size,
        )
        commands.append(command)


def run(cmd):
    subprocess.run(cmd)


with concurrent.futures.ThreadPoolExecutor(max_workers=-1) as executor:
    executor.map(run, commands)
