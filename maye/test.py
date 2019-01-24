import os

from flask import Flask, request, redirect, url_for, flash, render_template
from werkzeug.utils import secure_filename
from flask import send_from_directory

UPLOAD_FOLDER = '../web'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__, template_folder='./')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'files' not in request.files:
            print("Here")
            flash('No file part')
            return redirect(request.url)
        file = request.files['files']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print("HereHere")
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            t = ['b758f5366.jpg', '03d0be3ab.jpg', '7e5b193ce.jpg', 'bcb504542.jpg', 'b758f5366.jpg', '03d0be3ab.jpg']
            return render_template('index.html', file_name=filename, predictions=zip(t, t))
    return render_template('index.html')


@app.route('/results/<file_name>')
def results(file_name, predictions):
    return render_template('index.html', file_name=file_name, predictions=predictions)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


@app.route('/imgs/<filename>')
def training_imgs(filename):
    return send_from_directory('../data-train/',
                               filename)


app.run('0.0.0.0', 8888, debug=True)


