from bokeh.models.widgets.tables import HTMLTemplateFormatter
from flask import Flask, render_template, session
from flask.helpers import send_file
from flask_bootstrap import Bootstrap
from flask import request
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

from bokeh.resources import INLINE
from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, DataTable, TableColumn
from bokeh.models import LinearColorMapper, ColorBar, BasicTicker, PrintfTickFormatter
from bokeh.transform import transform
from bokeh.layouts import layout,column, Column

import pandas as pd
from sklearn.metrics import confusion_matrix

from flask_sqlalchemy import SQLAlchemy

import os
import io
import uuid

from torchvision import datasets
from PIL import Image

app = Flask(__name__)
app.secret_key = 'mD6v2NsJ'
bootstrap = Bootstrap(app)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///login.db'

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)

class User(UserMixin, db.Model):
    id = db.Column(db.String(40), primary_key=True)
    csv_data = db.Column(db.PickleType())
db.create_all()

# grab the static resources
js_resources = INLINE.render_js()
css_resources = INLINE.render_css()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(user_id)

testset = datasets.MNIST(root='./data', train=False, download=True)

@app.route('/mnist', methods=['GET'])
def mnist():
    index = int(request.args.get('index', '-1'))
    if index == -1:
        return "image not found", 404
    im = testset.data[index].detach().numpy()
    img = Image.fromarray(im)
    img_bin = io.BytesIO()
    img.save(img_bin, 'png')
    img_bin.seek(0)
    return send_file(img_bin, mimetype='image/png', as_attachment=False)

@app.route('/mnist/cm', methods=['GET'])
def mnist_cm():
    label = int(request.args.get('label', '-1'))
    if label == -1:
        return "label not found", 404
    pred = int(request.args.get('pred', '-1'))
    if pred == -1:
        return "pred not found", 404
    ##### セッションデータ読み込み開始
    user_id = session.get('user_id', str(uuid.uuid4()))
    session['user_id'] = user_id
    user = User.query.get(user_id)
    target_data = pd.DataFrame()
    if user is None:
        user = User(id=user_id, csv_data=pd.DataFrame().to_json())
        db.session.add(user)
        db.session.commit()

    try:
        target_data = pd.read_json(user.csv_data)
    except Exception as e:
        warn = str(e)

    df_columns = target_data.columns.tolist()
    # 正解ラベルカラムと推論ラベルカラムの読み込み
    label_column = session.get('label_column', '')
    pred_column = session.get('pred_column', '')
    cm = target_data[target_data[label_column]==label & target_data[pred_column]==pred]

    table_columns = []
    for column in cm.columns.tolist():
        table_columns.append(TableColumn(field=column, title=column))

    source = ColumnDataSource(data = cm)
    table =DataTable(source=source,selectable = True, columns=table_columns,
                sortable = True)
    layout_disp = table
    # render template
    script, div = components(layout_disp)
    return render_template(
        'index.html.j2',
        warn=warn,
        columns=df_columns,
        label_column=label_column,
        pred_column=pred_column,
        plot_script=script,
        plot_div=div,
        js_resources=js_resources,
        css_resources=css_resources,
    )

@app.route('/', methods=['GET', 'POST'])
def index():
    # 画面表示用変数のデフォルト値
    warn = ''

    # セッションクリアボタン押下時
    if 'clear' in request.form:
        for key in list(session.keys()):
            session.pop(key)
        session.clear()
        logout_user()

    ##### セッションデータ読み込み開始
    user_id = session.get('user_id', str(uuid.uuid4()))
    session['user_id'] = user_id
    user = User.query.get(user_id)
    target_data = pd.DataFrame()
    if user is None:
        user = User(id=user_id, csv_data=pd.DataFrame().to_json())
        db.session.add(user)
        db.session.commit()

    try:
        target_data = pd.read_json(user.csv_data)
    except Exception as e:
        warn = str(e)

    # 正解ラベルカラムと推論ラベルカラムの読み込み
    label_column = session.get('label_column', '')
    pred_column = session.get('pred_column', '')

    ##### 各フォーム変数読み込み開始

    # label,predカラム設定読み込み
    if 'label_column' in request.form:
        label_column = request.form['label_column']
        session['label_column'] = label_column
    if 'pred_column' in request.form:
        pred_column = request.form['pred_column']
        session['pred_column'] = pred_column

    # ファイルアップロード時の処理
    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            try:
                target_data = pd.read_csv(file)
                user.csv_data = target_data.to_json()
                db.session.commit()
            except Exception as e:
                warn = str(e)

    ##### 各画面要素作成
    # テーブル作成
    df_columns = target_data.columns.tolist()
    source = ColumnDataSource(data = target_data)

    table_columns = []
    for column in df_columns:
        table_columns.append(TableColumn(field=column, title=column))
    if 'index' in df_columns:
        table_columns.append(TableColumn(field='index', title='image',
            formatter=HTMLTemplateFormatter(template='<a href="./mnist?index=<%= value %>" "target=_blank"><img src="./mnist?index=<%= value %>"/></a>')))

    layout_disp = DataTable(source=source,selectable = True, columns=table_columns,
                sortable = True)

    # コンフュージョンマトリックス作成
    if label_column != '' and pred_column != '':
        cm = confusion_matrix(target_data[label_column], target_data[pred_column])
        length = len(cm)
        cm = pd.DataFrame(cm)
        cm['sum'] = cm.sum(axis='columns')
        cm.columns = [str(col) for col in cm.columns]
        cm = cm.append(cm.sum(axis='index'), ignore_index=True)

        source = ColumnDataSource(data = cm)
        # high = cm.max().max()
        # low = cm.min().min()
        # colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
        # mapper = LinearColorMapper(palette=colors, low=low, high=high)
        # p = figure(title="confusion matrix",
        #         toolbar_location=None, tools="", x_axis_location="above")
        # p.rect(x="pred", y="label", width=1, height=1, source=source,
        #     line_color=None, fill_color=transform('rate', mapper))
        # color_bar = ColorBar(color_mapper=mapper,
        #             ticker=BasicTicker(desired_num_ticks=len(colors)),
        #             formatter=PrintfTickFormatter(format="%d"))
        # p.add_layout(color_bar, 'right')
        # layout_disp = p
        # source = ColumnDataSource(data = cm)

        table_columns = []
        for column in cm.columns.tolist():
            table_columns.append(TableColumn(field=column, title=column,
                formatter=HTMLTemplateFormatter(template=f'<a href="./cm?label=<%= index %>&pred={column}" "target=_blank"><%= value %></a>')))

        table =DataTable(source=source,selectable = True, columns=table_columns,
                    sortable = True)
        layout_disp = Column(table, layout_disp)

    # render template
    script, div = components(layout_disp)
    return render_template(
        'index.html.j2',
        warn=warn,
        columns=df_columns,
        label_column=label_column,
        pred_column=pred_column,
        plot_script=script,
        plot_div=div,
        js_resources=js_resources,
        css_resources=css_resources,
    )

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
