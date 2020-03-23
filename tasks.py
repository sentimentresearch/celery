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
from urllib.request import urlretrieve


app = Celery('tasks', broker=os.getenv("CELERY_BROKER_URL", "redis://127.0.0.1:6379"))


@app.task
def bulk_predict(data, email, first_name, last_name):
    urlretrieve(os.getenv('MODEL_DROPBOX_LINK'), 'model_files.zip')
    zipfile.ZipFile('model_files.zip').extractall()

    args = {'use_multiprocessing': False, 'no_cache': True, 'use_cached_eval_features': False,
            'reprocess_input_data': True, 'silent': False}

    model = ClassificationModel('roberta', 'model_files/', use_cuda=False, args=args)

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
        file = f.read()
        f.close()

    encoded = base64.b64encode(file).decode()
    attachment = Attachment()
    attachment.file_content = FileContent(encoded)
    attachment.file_type = FileType('text/csv')
    attachment.file_name = FileName(filename)
    attachment.disposition = Disposition('attachment')
    message.attachment = attachment

    client = SendGridAPIClient(os.getenv("SENDGRID_KEY"))
    client.send(message)

    message = Mail(
        from_email='thilo@colabel.com',
        to_emails='sentiment-research.bwl@uni-hamburg.de',
        subject='New submission (sentiment model)',
        html_content='Number of rows: ' + str(len(data))
    )
    client.send(message)

    os.remove(filename)
