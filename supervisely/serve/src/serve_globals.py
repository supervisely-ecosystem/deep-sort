import os
import sys
import pathlib
import supervisely_lib as sly
from dotenv import load_dotenv  # pip install python-dotenv\

load_dotenv("../debug.env")
load_dotenv("../secret_debug.env", override=True)

my_app = sly.AppService()
api = my_app.public_api
task_id = my_app.task_id

logger = sly.logger

sly.fs.clean_dir(my_app.data_dir)  # @TODO: for debug

root_source_path = str(pathlib.Path(sys.argv[0]).parents[3])
sly.logger.info(f"Root source directory: {root_source_path}")
sys.path.append(root_source_path)

serve_source_path = os.path.join(root_source_path, "supervisely/serve/")
sly.logger.info(f"Serve source directory: {serve_source_path}")
sys.path.append(serve_source_path)

UI_source_path = os.path.join(root_source_path, "supervisely/serve/src/ui")
sly.logger.info(f"UI source directory: {serve_source_path}")
sys.path.append(serve_source_path)

owner_id = int(os.environ['context.userId'])
team_id = int(os.environ['context.teamId'])
workspace_id = int(os.environ['context.workspaceId'])

model_info = None
model_meta = None

opt = None

selected_classes = []


def get_files_paths(src_dir, extensions):
    files_paths = []
    for root, dirs, files in os.walk(src_dir):
        for extension in extensions:
            for file in files:
                if file.endswith(extension):
                    file_path = os.path.join(root, file)
                    files_paths.append(file_path)

    return files_paths


def finish_step(step_num):
    next_step = step_num + 1
    fields = [
        {"field": f"data.done{step_num}", "payload": True},
        {"field": f"state.collapsed{next_step}", "payload": False},
        {"field": f"state.disabled{next_step}", "payload": False},
        {"field": f"state.activeStep", "payload": next_step},
    ]
    api.app.set_field(task_id, "data.scrollIntoView", f"step{next_step}")
    api.app.set_fields(task_id, fields)

