import os
from celery import Celery
import base64
from sendgrid.helpers.mail import (
    Mail, Attachment, FileContent, FileName,
    FileType, Disposition)
from sendgrid import SendGridAPIClient
from uuid import uuid4
import csv
from simpletransformers.classification import ClassificationModel
import zipfile
import aiohttp
import asyncio

app = Celery('tasks', broker=os.getenv("CELERY_BROKER_URL", "redis://127.0.0.1:6379"))

export_file_url = 'https://www.dropbox.com/s/bgljpmn90u7v91n/model_files.zip?raw=1'
export_file_name = 'model_files.zip'


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
    classifier = ClassificationModel('roberta', 'model_files/', use_cuda=False, args=args)
    return classifier


@app.task
def bulk_predict(data, email, first_name, last_name):
    loop = asyncio.get_event_loop()
    tasks = [asyncio.ensure_future(setup_model())]
    model = loop.run_until_complete(asyncio.gather(*tasks))[0]
    loop.close()

    predictions = model.predict(data)[0].tolist()

    message = Mail(
        from_email='thilo@colabel.com',
        to_emails=email,
        subject='Your sentiment analysis results are ready',
        html_content='Dear ' + first_name + ' ' + last_name + ' you can find your data attached to this email.'
    )

    filename = str(uuid4()) + '.csv'
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'label'])
        for text, prediction in zip(data, predictions):
            writer.writerow([text, prediction])

    with open(filename, 'rb') as f:
        data = f.read()
        f.close()

    encoded = base64.b64encode(data).decode()
    attachment = Attachment()
    attachment.file_content = FileContent(encoded)
    attachment.file_type = FileType('text/csv')
    attachment.file_name = FileName(filename)
    attachment.disposition = Disposition('attachment')
    message.attachment = attachment

    client = SendGridAPIClient(os.getenv("SENDGRID_KEY"))
    client.send(message)

    os.remove(filename)
