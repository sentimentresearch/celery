import uvicorn
from io import StringIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import UJSONResponse
from starlette.templating import Jinja2Templates
from tasks import bulk_predict
import csv
import zipfile
from urllib.request import urlretrieve
from simpletransformers.classification import ClassificationModel
import os
import numpy as np


app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['*'], allow_methods=['*'])
templates = Jinja2Templates(directory='templates')

urlretrieve(os.getenv('MODEL_DROPBOX_LINK'), 'model_files.zip')
zipfile.ZipFile('model_files.zip').extractall()

args = {'use_multiprocessing': False, 'no_cache': True, 'use_cached_eval_features': False,
            'reprocess_input_data': True, 'silent': False}

model = ClassificationModel('roberta', 'model_files/', use_cuda=False, args=args)

label_mapping = {"0": 0, "1": 1}


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


@app.route('/predict', methods=['POST'])
async def predict(request):
    body = await request.json()
    data = body['data']

    instances = model.predict(data)[1].tolist()

    results = []
    for i, instance_scores in enumerate(instances):

        instance_scores = list(softmax(instance_scores))

        predictions = []
        for j, score in enumerate(instance_scores):
            predictions.append({'class': label_mapping[str(j)], 'score': score})
        predictions = sorted(predictions, key=lambda k: k['score'], reverse=True)

        results.append({'text': data[i], 'predictions': predictions})

    return UJSONResponse({'results': results})


@app.route('/bulk-predict', methods=['POST'])
async def bulk_prediction(request):
    form = await request.form()
    contents = await form['file'].read()

    reader = csv.reader(StringIO(contents.decode()))
    data = []
    for row in reader:
        if len(row) > 0:
            data.append(row[0])

    data = data[:100000]

    email = form['email']
    first_name = form['first-name']
    last_name = form['last-name']

    bulk_predict.delay(data, form['file'].filename, email, first_name, last_name)

    return templates.TemplateResponse('thank-you.html', {'request': request})


@app.route("/")
def bulk_form(request):
    return templates.TemplateResponse('index.html', {'request': request})


if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
