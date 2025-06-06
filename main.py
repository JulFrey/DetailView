from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import threading
import uuid
import time
import predict  # your predict.py

app = FastAPI()
task_store: Dict[str, Dict] = {}

class PredictRequest(BaseModel):
    prediction_data: str = r"/input/mini.las"
    path_las: str = ""
    model_path: str = "./model_ft_202412171652_3"
    tree_id_col: str = "TreeID"
    n_aug: int = 10

@app.post("/predict/start")
def start_prediction(req: PredictRequest):
    task_id = str(uuid.uuid4())
    task_store[task_id] = {"status": "running", "result": None}

    def run_task():
        try:
            outfile, outfile_probs, joined, data_probs_df = predict.run_predict(
                prediction_data=req.prediction_data,
                path_las=req.path_las,
                model_path=req.model_path,
                tree_id_col=req.tree_id_col,
                n_aug=req.n_aug
            )
            task_store[task_id] = {
                "status": "completed",
                "result": {
                    "outfile": outfile,
                    "outfile_probs": outfile_probs,
                    "joined": joined.to_dict(orient="records"),
                    "data_probs": data_probs_df.to_dict(orient="records")
                }
            }
        except Exception as e:
            task_store[task_id] = {"status": "failed", "error": str(e)}

    threading.Thread(target=run_task).start()

    return {"task_id": task_id}

@app.get("/predict/status/{task_id}")
def get_status(task_id: str):
    task = task_store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task ID not found")
    return {"status": task["status"]}

@app.get("/predict/result/{task_id}")
def get_result(task_id: str):
    task = task_store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task ID not found")
    if task["status"] != "completed":
        raise HTTPException(status_code=202, detail="Task not yet completed")
    return task["result"]
