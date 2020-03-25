import os
from celery import Celery
import base64
from sendgrid.helpers.mail import (
    Mail, Attachment, FileContent, FileName,
    FileType, Disposition)
from sendgrid import SendGridAPIClient
import csv
from simpletransformers.classification import ClassificationModel
import zipfile
from urllib.request import urlretrieve
import numpy as np


app = Celery('tasks', broker=os.getenv("CELERY_BROKER_URL", "redis://127.0.0.1:6379"))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


@app.task
def bulk_predict(data, original_file_name, email, first_name, last_name):
    urlretrieve(os.getenv('MODEL_DROPBOX_LINK'), 'model_files.zip')
    zipfile.ZipFile('model_files.zip').extractall()

    args = {'use_multiprocessing': False, 'no_cache': True, 'use_cached_eval_features': False,
            'reprocess_input_data': True, 'silent': False}

    model = ClassificationModel('roberta', 'model_files/', use_cuda=False, args=args)

    predictions = model.predict(data)[1].tolist()

    message = Mail(
        from_email='sentiment@colabel.com',
        to_emails=email,
        subject='Your sentiment analysis results are ready',
        html_content='Dear ' + first_name + ' ' + last_name + ',<br><br>Please find your data with sentiment predictions attached to this email.'
                        '<br><br>Thanks for using our service. We appreciate any feedback you may have.'
                        '<br><br>For full details on accuracy and sentiment analysis benchmarks please download our full paper here:'
                        '<br>www.accurate-sentiment-analysis.com'
                        '<br><br>Was this service useful for you? If so, please support us by spreading the word and sharing our website with anyone potentially interested.'
                        '<br><br>Kind regards,'
                        '<br>Mark Heitmann, Christian Siebert, Jochen Hartmann, and Christina Schamp'
                        '<br><br>sentiment-research.bwl@uni-hamburg.de'
    )

    filename = original_file_name.replace('.csv', '_wSentiment.csv')
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'negative', 'positive'])
        for text, prediction in zip(data, predictions):
            writer.writerow([text, *list(softmax(prediction))])

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
        from_email='sentiment@colabel.com',
        to_emails='sentiment-research.bwl@uni-hamburg.de',
        subject='New submission (sentiment model)',
        html_content='Number of rows: ' + str(len(data))
    )
    client.send(message)

    os.remove(filename)

# TODO
# sendgrid smtp (fwd answers?)
# Limit rows to 100k client-side - red/invalid field + notification
# save all columns in output file
# make column selection work (papaparse & bulma)
# detect delimiter
# make first and list splitted (Bulma)
# ramp up celery & redis machines
# (validate text column)
