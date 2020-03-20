import aiohttp
import asyncio
import uvicorn
from io import StringIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import UJSONResponse, RedirectResponse
from simpletransformers.classification import ClassificationModel
import zipfile
import os
from starlette.templating import Jinja2Templates
from tasks import bulk_predict
import csv

export_file_url = os.getenv('MODEL_DROPBOX_LINK')
export_file_name = 'model_files.zip'


app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['*'], allow_methods=['*'])
templates = Jinja2Templates(directory='templates')


async def download_file(url, dest):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_model():
    await download_file(export_file_url, export_file_name)
    args = {'use_multiprocessing': False, 'no_cache': True, 'use_cached_eval_features': False,
            'reprocess_input_data': True, 'silent': True}
    zipfile.ZipFile('model_files.zip').extractall()
    model = ClassificationModel('roberta', 'model_files/', use_cuda=False, args=args)
    return model


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_model())]
model = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

label_mapping = {"0": 0, "1": 1}


@app.route('/predict', methods=['POST'])
async def predict(request):
    body = await request.json()
    data = body['data']

    instances = model.predict(data)[1].tolist()

    results = []
    for i, instance_scores in enumerate(instances):

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
        data.append(row[0])

    email = form['email']
    first_name = form['first_name']
    last_name = form['last-name']

    bulk_predict.delay(data, email, first_name, last_name)

    return RedirectResponse('/')


@app.route("/")
def form(request):
    return templates.TemplateResponse('index.html', {'request': request})


if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
