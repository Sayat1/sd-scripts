from pathlib import Path
import os
import modal
# Load the .env file if it exists


# must come before ANY torch or fastai imports
# import toolkit.cuda_malloc

# turn off diffusers telemetry until I can figure out how to make it opt-in

model_volume = modal.Volume.from_name("myvolume", create_if_missing=True)
MOUNT_DIR = "/root/output/"  # modal_output, due to "cannot mount volume on non-empty path" requirement
image = (
        modal.Image.from_registry("nvidia/cuda:12.9.0-base-ubuntu24.04", add_python="3.11")
        # install required system and pip packages, more about this modal approach: https://modal.com/docs/examples/dreambooth_app
        .apt_install(
            "libgl1",
            "libglib2.0-0",
            "git",
            "build-essential",
            "clang",
            "pkg-config"
        )
        .run_commands(
            "pip install --upgrade pip",
            "cd /root && git clone -b 'sd3' https://github.com/Sayat1/sd-scripts",
            "pip install -r /root/sd-scripts/requirements.txt",
            "pip install -U prodigy-plus-schedule-free",
            "pip install -U lycoris-lora",
            "pip install hf_transfer",
            "pip install torchvision"
        )
    )


# create the Modal app with the necessary mounts and volumes
app = modal.App(name="modal-training")


@app.function(
    # request a GPU with at least 24GB VRAM
    # more about modal GPU's: https://modal.com/docs/guide/gpu
    gpu="H100", # gpu="H100" L40S
    # more about modal timeouts: https://modal.com/docs/guide/timeouts
    timeout=3600*3,  # 2 hours, increase or decrease if needed,
    image=image, 
    volumes={MOUNT_DIR: model_volume}
)
def remote_main(args):
    print("Running modal with arguments:", args)
    import subprocess, os, sys

    from accelerate.utils import write_basic_config
    write_basic_config(mixed_precision="bf16")
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    sys.path.insert(0, "/root/sd-scripts")
    os.environ['DISABLE_TELEMETRY'] = 'YES'
    # Check if we have DEBUG_TOOLKIT in env
    if os.environ.get("DEBUG_TOOLKIT", "0") == "1":
        # Set torch to trace mode
        import torch
        torch.autograd.set_detect_anomaly(True)


    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    def _exec_subprocess(cmd: list[str]):
        """Executes subprocess and prints log to terminal while subprocess is running."""
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
            env=env
        )
        with process.stdout as pipe:
            for line in iter(pipe.readline, b""):
                line_str = line
                print(f"{line_str}", end="")

        if exitcode := process.wait() != 0:
            raise subprocess.CalledProcessError(exitcode, "\n".join(cmd))
        
    print("launching dreambooth training script")
    os.chdir("/root/sd-scripts")

    import sdxl_train_network

    sys.argv = args
    sdxl_train_network.main()

    # cmd_args = [
    #         "accelerate",
    #         "launch"
    # ] + args
    # _exec_subprocess(
    #     cmd_args
    # )
    model_volume.commit()


def setup_toml(toml_path: Path) -> dict:
    import toml
    imagesets = {}
    dataset_count = 0
    local_toml = toml.load(toml_path)
    for dataset in local_toml["datasets"]:
        for subset in dataset["subsets"]:
            if "image_dir" in subset:
                if subset["image_dir"] not in imagesets.values():
                    imagesets[dataset_count] = subset["image_dir"]
                    subset["image_dir"] = MOUNT_DIR + f"/dataset/{dataset_count}"
                    dataset_count += 1
                else:
                    for key, value in imagesets.items():
                        if value == subset["image_dir"]:
                            subset["image_dir"] = MOUNT_DIR + f"/dataset/{key}"

    with open("dataset_modal.toml",'w',encoding="utf-8") as f:
        toml.dump(local_toml,f)
    return imagesets

def is_probably_path(s: str) -> bool:
    p = Path(s)
    return (
        "/" in s or "\\" in s
    )

@app.local_entrypoint()
def local_main(commendtxt: str):
    global image

    print("local entrypoint")

    output_dir = "/root/output/"
    log_dir = Path("root/output/")
    checkpoint_filepath = ""
    dataset_filepath = ""
    lycoris_preset_path = ""
    cmd_list_str = ""
    print(commendtxt)
    if os.path.exists(commendtxt):
        with open(commendtxt, "r", encoding="utf-8") as f:
            cmd_list_str = f.read()
    else:
        raise FileNotFoundError(f"Command text file '{commendtxt}' not found.")
    args_list = cmd_list_str.split()
    print(args_list)
    for i, arg in enumerate(args_list):
        if 'output_dir' in str(arg):
            args_list[i + 1] = output_dir
        elif 'logging_dir' in str(arg):
            args_list[i + 1] = str((log_dir.joinpath(*Path(args_list[i + 1]).parts[-2:])).as_posix())
        elif 'pretrained_model_name_or_path' in str(arg):
            checkpoint_filepath = args_list[i + 1]
            args_list[i + 1] = str(MOUNT_DIR + Path(checkpoint_filepath).name)
        elif 'dataset_config' in str(arg):
            dataset_filepath = Path(args_list[i + 1])
            args_list[i + 1] = MOUNT_DIR + "dataset_modal.toml"
        elif 'preset=' in str(arg):
            preset_path = arg[6:]
            if is_probably_path(preset_path):
                lycoris_preset_path = preset_path
                args_list[i] = f"preset={str(MOUNT_DIR + Path(preset_path).name)}"

    print(args_list)
    local_dataset_dirs_dict = setup_toml(dataset_filepath)
    
    upload_chekkpoint = True
    for file_entry in model_volume.iterdir(path="/", recursive=False):
        print(file_entry.path)
        if Path(checkpoint_filepath).name in file_entry.path:
            print("Found existing checkpoint in volume, skipping upload")
            upload_chekkpoint = False
            break

    if upload_chekkpoint:
        with model_volume.batch_upload() as batch:
            batch.put_file(checkpoint_filepath, Path(checkpoint_filepath).name)

    with model_volume.batch_upload(force=True) as batch:
        batch.put_file("dataset_modal.toml", "dataset_modal.toml")
        
        for index, dataset_dir in local_dataset_dirs_dict.items():
            batch.put_directory(dataset_dir, "dataset/" + str(index))
        if lycoris_preset_path != "":
            batch.put_file(lycoris_preset_path, Path(lycoris_preset_path).name)

    remote_main.remote(args_list)

if __name__ == "__main__":
    import sys
    local_main(" ".join(sys.argv[1:]))