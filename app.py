from threading import Thread
from flask import Flask, render_template, session
from flask import request
from flask.helpers import send_file
from tornado.ioloop import IOLoop
from flask_login import LoginManager, UserMixin

from bokeh.server.server import Server
from bokeh.embed import server_document

from bokeh.models.widgets.tables import HTMLTemplateFormatter
from bokeh.models import ColumnDataSource, DataTable, TableColumn, LinearColorMapper, ColorBar, CDSView, BooleanFilter
from bokeh.models.widgets import Button, FileInput, Select, MultiSelect, RadioGroup
from bokeh.events import ButtonClick, DoubleTap
from bokeh.layouts import Column, Row
from bokeh.plotting import figure
from bokeh.transform import transform, linear_cmap
from bokeh.palettes import Viridis256

from flask_sqlalchemy import SQLAlchemy

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import io
import base64
import uuid

from torchvision import datasets
from PIL import Image

app = Flask(__name__)
app.secret_key = 'mD6v2NsJ'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)

class User(UserMixin, db.Model):
    id = db.Column(db.String(40), primary_key=True)
    csv_data = db.Column(db.PickleType())
db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(user_id)

testset = datasets.MNIST(root='./data', train=False, download=True)

target_data = pd.DataFrame()
user = None

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

def create_datatable_columns(df_columns):
    table_columns = []
    for column in df_columns:
        table_columns.append(TableColumn(field=column, title=column))
    if 'index' in df_columns:
        table_columns.append(TableColumn(field='index', title='image',
            formatter=HTMLTemplateFormatter(template='<a href="./mnist?index=<%= value %>" target="_blank"><img src="./mnist?index=<%= value %>"/></a>')))
    return table_columns

def create_disp(label_column='', pred_column=''):
    ##### 各画面要素作成
    # テーブル作成
    df_columns = target_data.columns.tolist()
    source = ColumnDataSource(data = target_data)
    table_columns = create_datatable_columns(df_columns)

    data_table = DataTable(source=source,selectable = True, columns=table_columns,
                sortable = True)

    # 正解ラベル
    label_select = Select(title="正解ラベル:", value="", options=[""])
    # 推論ラベル
    pred_select = Select(title="推論ラベル:", value="", options=[""])

    # 散布図x
    x_select = Select(title="散布図x:", value="", options=[""])
    # 散布図y
    y_select = Select(title="散布図y:", value="", options=[""])
    # 色カラム
    color_select = Select(title="色カラム:", value="", options=[""])

    # 特徴量フィールド選択
    dim_reduction = RadioGroup(labels=['None', 'PCA', 'tSNE'], active=0)
    feature_select = MultiSelect()

    # upload button がクリックされた時の処理　
    def upload_button_callback(event):
        global target_data
        global user
        csv_str = base64.b64decode(csv_input.value)
        target_data = pd.read_csv(io.BytesIO(csv_str))
        df_columns = target_data.columns.tolist()
        label_select.options = df_columns
        pred_select.options = df_columns
        x_select.options = df_columns
        y_select.options = df_columns
        color_select.options = df_columns
        table_columns = create_datatable_columns(df_columns)
        data_table.columns = table_columns
        source.data = target_data
        data_table.update()
        try:
            user.csv_data = target_data.to_json()
            db.session.commit()
        except Exception as e:
            warn = str(e)
        feature_select.options = target_data.columns.tolist()

    # csvアップロード実行ボタン
    upload_button = Button(label="Upload", button_type="success")
    upload_button.on_event(ButtonClick, upload_button_callback)

    # ファイル選択ボックス
    csv_input = FileInput()

    # 混同行列ヒートマップ
    confusion_matrix_source = ColumnDataSource(data = {})

    tooltips=[
        ( 'count',   '@count' ),
        ( 'label',   '$y{0}' ),
        ( 'pred',   '$x{0}' ),
    ]
    hm = figure(title="混同行列", tools="hover", toolbar_location=None, tooltips=tooltips)
    hm.y_range.flipped=True
    hm.visible = False

    dim_reduction_plot = figure(title="特徴量空間", tools="zoom_in,wheel_zoom,box_zoom,hover,box_select,poly_select,lasso_select,tap,reset")
    dim_reduction_plot.visible = False

    def cm_callback():
        global target_data
        cm = confusion_matrix(target_data[label_select.value], target_data[pred_select.value])
        cm = pd.DataFrame(cm)
        cm = cm.stack().reset_index().rename(columns={'level_0':label_select.value, 'level_1':pred_select.value,0:'count'})
        confusion_matrix_source.data = cm

        colors = ['#75968f', '#a5bab7', '#c9d9d3', '#e2e2e2', '#dfccce', '#ddb7b1', '#cc7878', '#933b41', '#550b1d']
        mapper = LinearColorMapper(palette=colors, low=cm['count'].min(), high=cm['count'].max())
        hm.rect(x=pred_select.value, y=label_select.value, source=confusion_matrix_source, width=1, height=1, line_color=None, fill_color=transform('count', mapper))
        hm.text(x=pred_select.value, y=label_select.value, text="count", text_font_style="bold", source=confusion_matrix_source,
                text_align= "left", text_baseline="middle")
        
        color_bar = ColorBar(color_mapper=mapper, label_standoff=12)
        hm.add_layout(color_bar, 'right')

        hm.visible = True
        dim_reduction_plot.visible = False

    def cm_click_callback(event):
        label = round(event.y)
        pred = round(event.x)
        data = target_data[(target_data[label_select.value]==label) & (target_data[pred_select.value]==pred)]
        source.data = data

    hm.on_event(DoubleTap, cm_click_callback)

    scatter_source = ColumnDataSource()

    def on_dim_change(self, attr, *callbacks):
        print(scatter_source.selected.indices)
#        data = target_data[scatter_source.selected.indices]
        data = target_data.loc[scatter_source.selected.indices,:]
        print(data)
        source.data = data

    def dim_reduction_change():
        global target_data
        col = color_select.value
        scatter_source.data = {col: target_data[col], 'legend': [f'{col}_{x}' for x in target_data[col]], 'x':target_data[x_select.value], 'y': target_data[y_select.value]}
        mapper = linear_cmap(field_name=col, palette=Viridis256,
                            low=min(target_data[col].values), high=max(target_data[col].values))
        plot = dim_reduction_plot.circle(x="x", y="y", source=scatter_source, line_color=mapper, color=mapper, legend_field='legend')
        plot.data_source.selected.on_change('indices', on_dim_change)
        dim_reduction_plot.legend.location = "top_right"
        dim_reduction_plot.legend.click_policy = "hide"

        hm.visible = False
        dim_reduction_plot.visible = True

    # 混同行列作成実行ボタン
    cm_button = Button(label="混同行列作成", button_type="success")
    cm_button.on_event(ButtonClick, cm_callback)

    # 散布図作成実行ボタン
    scatter_button = Button(label="散布図作成", button_type="success")
    scatter_button.on_event(ButtonClick, dim_reduction_change)

    # 混同行列オペレーションエリア
    cm_operation_area = Column(label_select, pred_select, cm_button)
    # 散布図オペレーションエリア
    scatter_operation_area = Column(x_select, y_select, color_select, scatter_button)
    # サブオペレーションエリア
    sub_operation_area = Row(cm_operation_area, scatter_operation_area)

    operation_area = Column(csv_input, upload_button, sub_operation_area, data_table)
    graph_area = Column(hm, dim_reduction_plot)
    layout_disp = Row(graph_area, operation_area)

    return layout_disp

def bkapp(doc):
    doc.add_root(create_disp())
    doc.title = "mnist-viewer"

@app.route('/', methods=['GET'])
def bkapp_page():
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

    script = server_document('http://localhost:5006/bkapp')
    return render_template("embed.html", script=script, template="Flask")

def bk_worker():
    server = Server({'/bkapp': bkapp}, io_loop=IOLoop(), allow_websocket_origin=["127.0.0.1:8080"])
    server.start()
    server.io_loop.start()

Thread(target=bk_worker).start()

if __name__ == '__main__':
    app.run(port=8080)
