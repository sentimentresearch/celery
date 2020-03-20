import uvicorn
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.templating import Jinja2Templates
from starlette.responses import RedirectResponse

from tasks import bulk_predict

import csv
from io import StringIO

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['*'], allow_methods=['*'])
templates = Jinja2Templates(directory='templates')


@app.route("/")
def form(request):
    return templates.TemplateResponse('index.html', {'request': request})


@app.route('/bulk-predict', methods=['POST'])
async def bulk_prediction(request):
    form = await request.form()
    contents = await form['file'].read()

    reader = csv.reader(StringIO(contents.decode()))
    data = []
    for row in reader:
        data.append(row[0])

    print(form['email'])

    return RedirectResponse('/')


if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")