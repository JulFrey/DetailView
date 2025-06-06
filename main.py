from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
import predict

class PredictRequest(BaseModel):
    prediction_data: str = r"/input/mini.las"
    path_las: str = ""
    model_path: str = "./model_ft_202412171652_3"
    tree_id_col: str = "TreeID"
    n_aug: int = 10

app = FastAPI()

@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    try:
        outfile, outfile_probs, joined, data_probs_df = predict.run_predict(
            prediction_data=req.prediction_data,
            path_las=req.path_las,
            model_path=req.model_path,
            tree_id_col=req.tree_id_col,
            n_aug=req.n_aug
        )

        # Convert pandas DataFrames to JSON-serializable structures
        joined_json = joined.to_dict(orient="records")
        probs_json = data_probs_df.to_dict(orient="records")

        return JSONResponse(content={
            "outfile": outfile,
            "outfile_probs": outfile_probs,
            "joined": joined_json,
            "data_probs": probs_json
        })

    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")
