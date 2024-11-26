import os
import json
import time
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_file, flash, session, jsonify
from dataprepro.apps.utils.feature_selection_algorithm import FeatureSelectionAlgorithm
from wtforms.fields.simple import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired
from dataprepro.apps.utils.datacleaning import DataCleaner
from flask_login import LoginManager, login_required, logout_user
from werkzeug.utils import secure_filename
from dataprepro.apps.settings.settings_manager import SettingManagers
from dataprepro.apps.users.user import User
from dataprepro.apps.utils.captcha_generator import CaptchaGenerator
from typing import Optional
import pandas as pd
from dataprepro.apps.utils.login import Login
from flask_wtf import CSRFProtect, FlaskForm
from pandas import DataFrame


class LoginForm(FlaskForm):
    username = StringField('账号', validators=[DataRequired()])
    password = PasswordField('密码', validators=[DataRequired()])
    captcha = StringField('验证码', validators=[DataRequired()])


class DataPrePro:
    def __init__(self):
        self.app = Flask(__name__, template_folder='apps/templates', static_folder='apps/static')
        self.settings = SettingManagers()
        self.app.config['UPLOAD_FOLDER'] = self.settings.get('UPLOAD_FOLDER')  # 设置上传文件的保存目录
        self.app.config['MAX_CONTENT_LENGTH'] = self.settings.get('MAX_CONTENT_LENGTH')  # 最大允许上传文件大小 500MB
        self.app.secret_key = self.settings.get('SECRET_KEY', 'default_secret_key_12345')
        self.upload_path = []
        self.cleaned_path = []
        # Flask-Login 配置
        self.login_manager = LoginManager()
        self.login_manager.init_app(self.app)
        self.login_manager.login_view = self.settings.get('LOGIN_VIEW', 'login')
        self.login_manager.unauthorized_handler(self.unauthorized)  # 自定义未登录处理
        self.login: Optional[Login] = None
        self.cleaner: Optional[DataCleaner] = None
        self.left_cleaner: Optional[DataCleaner] = None
        self.right_cleaner: Optional[DataCleaner] = None
        self.mergeData: Optional[DataFrame] = None
        self.feature_select: Optional[FeatureSelectionAlgorithm] = None
        self.feature_sort = None
        self.register_routes()
        # 初始化 CSRF 保护
        csrf = CSRFProtect()
        csrf.init_app(self.app)

    # def allowed_file(self, filename):
    #     return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.app.config.get('ALLOWED_EXTENSIONS')

    def auto_field_matching(self, left_df, right_df, columns_left, columns_right, file_left_name, file_right_name):
        result = {'table_name': [file_left_name, file_right_name], 'left': [], 'right': []}
        commons = set(columns_left) & set(columns_right)
        for common in commons:
            if left_df[common].dtype == right_df[common].dtype:
                result['left'].append(common)
                result['right'].append(common)
        diff_left = set(columns_left) - set(result['left'])
        if diff_left:
            result['left'].extend(diff_left)
            result['right'].extend([None] * len(diff_left))
        left_on = [result['left'][i] for i in range(len(result['left'])) if (result['left'][i] and result['right'][i])]
        right_on = [result['right'][i] for i in range(len(result['right'])) if
                    (result['left'][i] and result['right'][i])]
        self.mergeData = self.merge_data(self.left_cleaner.data, self.right_cleaner.data, left_on=left_on,
                                         right_on=right_on)
        result['columns'] = list(self.mergeData.columns)
        merge_data = self.mergeData.replace({np.nan: None})
        result['rows'] = merge_data.values.tolist()
        diff_right = set(columns_right) - set(result['right'])
        if diff_right:
            result['left'].extend([None] * len(diff_right))
            result['right'].extend(diff_right)
        return jsonify(result), 200

    def merge_data(self, df1, df2, left_on=None, right_on=None):
        return pd.merge(df1, df2, left_on=left_on, right_on=right_on, how='outer')

    @staticmethod
    def unauthorized():
        flash('请先登录')
        return redirect(url_for('login'))

    def register_routes(self):
        self.login = Login()

        # Flask-Login 加载用户

        @self.login_manager.user_loader
        def load_user(user_id):
            return User(user_id)

        @self.app.route('/get_code')
        def get_code():
            fg_color = self.login.random_color()
            img_io, _, unique_id = CaptchaGenerator(fg_color=fg_color).create_validate_code()
            self.login.unique_id = unique_id
            return send_file(img_io, mimetype='image/png')

        @self.app.route('/', methods=['GET', 'POST'])
        def login():
            form = LoginForm()
            if form.validate_on_submit():
                return self.login.handle_login(form)
            else:
                print(form.errors)
            return render_template('login.html', form=form)

        @self.app.route('/handle_merge_data', methods=['POST'])
        @login_required
        def handle_merge_data():
            result = {}
            data = request.json
            left_row_data = data.get('leftRowData')
            right_row_data = data.get('rightRowData')
            merge_columns = data.get('mergeCloumns')
            result['columns'] = merge_columns
            left_on = [left_row_data[i] for i in range(len(left_row_data)) if
                       (left_row_data[i] != '空' and right_row_data[i] != '空')]
            right_on = [right_row_data[i] for i in range(len(right_row_data)) if
                        (left_row_data[i] != '空' and right_row_data[i] != '空')]
            for index in range(len(left_on)):
                if self.right_cleaner and self.left_cleaner and self.left_cleaner.data[left_on[index]].dtype == \
                        self.right_cleaner.data[right_on[index]]:  # noqa
                    continue
                else:
                    return jsonify({'success': False, 'message': '请正确完成合并依据'})
            try:
                self.mergeData = self.merge_data(self.left_cleaner.data, self.right_cleaner.data, left_on=left_on,
                                                 right_on=right_on)
            except Exception as e:
                return jsonify({'success': False, 'message': '请正确完成合并依据'})
            merge_data = self.mergeData.replace({np.nan: None})
            result['columns'] = list(merge_data.columns)
            result['rows'] = merge_data.values.tolist()
            result['success'] = True
            return jsonify(result), 200

        @self.app.route('/uploadLeft', methods=['POST'])
        @login_required
        def uploadLeft():
            file = request.files['file']
            if not file:
                return jsonify({'success': False, 'message': '未上传文件！'})
            try:
                session['right_initialized'] = True  # 标记右文件初始化成功
                return jsonify({'success': True, 'message': '右文件已成功初始化'})
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)})

        @self.app.route('/uploadRight', methods=['POST'])
        @login_required
        def uploadRight():
            file = request.files['file']
            if not file:
                return jsonify({'success': False, 'message': '未上传文件！'})
            try:
                session['right_initialized'] = True  # 标记右文件初始化成功
                return jsonify({'success': True, 'message': '右文件已成功初始化'})
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)})

        @self.app.route('/upload', methods=['POST'])
        @login_required
        def upload_file():
            if not request.files:
                flash('没有文件部分')
                return redirect(request.url)

            file = request.files['file']
            output_path = request.form.get('path')
            if not file:
                flash('没有选择文件')
                return redirect(request.url)
            if file.mimetype not in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                     'text/csv']:
                flash('文件格式不支持，请上传 XLSX 或 CSV 格式的文件。')
                return redirect(request.url)

            filename = secure_filename(file.filename)

            if file.content_length > 500 * 1024 * 1024:  # 限制文件大小为 500MB
                return jsonify({'success': False, 'message': '文件超出大小限制，请确保文件小于 500MB。!'})
            if output_path == '':
                return jsonify({'success': False, 'message': '请选择要保存的文件路径!'})
            if not os.path.exists(output_path) or os.path.isfile(output_path):
                return jsonify({'success': False, 'message': '请选择正确的文件路径!'})
            try:
                output_path += '\\upload'
                if output_path not in self.upload_path:
                    self.upload_path.append(output_path)
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                file_path = os.path.join(output_path, filename)
                file.save(file_path)  # 保存文件到指定目录
                flash(f'文件已成功保存至 {output_path}', 'success')
                return jsonify({'success': True, 'message': '文件上传成功！'})  # 返回成功响应
            except Exception as e:
                flash('文件上传失败，请重试。')
                return jsonify({'success': False, 'message': f'{e}!'})

        @self.app.route('/preview', methods=['POST'])
        @login_required
        def preview():
            if 'file' not in request.files:
                return jsonify({'error': 'No file provided'}), 400
            file = request.files['file']
            self.cleaner = DataCleaner(file=file)
            return self.cleaner.pre_view()

        @self.app.route('/index')
        @login_required
        def index():
            return render_template('index.html')

        @self.app.route('/logout', methods=['GET', 'POST'])
        @login_required
        def logout():
            logout_user()  # 登出用户
            flash('登出成功')
            return redirect(url_for('login'))

        @self.app.route('/upload_data', methods=['GET', 'POST'])
        @login_required
        def upload_data():

            return render_template('upload_data.html')

        # 处理数据清洗的各个步骤
        @self.app.route('/data_cleaning', methods=['GET', 'POST'])
        @login_required
        def data_cleaning():
            if request.method == 'GET':
                # 获取上传文件夹中的所有文件
                uploaded_files = {}
                for path in self.upload_path:
                    if os.path.exists(path):
                        for file in os.listdir(path):
                            uploaded_files[f"{path}\\{file}"] = file
                return render_template('data_cleaning.html', step='file_selection', files=uploaded_files)

            # 获取表单中用户选择的文件
            selected_file = request.form.get('selected_file')
            if selected_file:
                self.cleaner = DataCleaner(file_path=selected_file)
                self.cleaner.load_data()
                all_columns = self.cleaner.get_columns()
                if len(all_columns) == len(set(all_columns)):
                    return render_template('data_cleaning.html', step='column_duplicate_removal', columns=all_columns)
                else:
                    flash('该文件存在相同的列名，不符合')
                    return redirect(url_for('data_cleaning'))
            flash('请选择一个文件')
            return redirect(url_for('data_cleaning'))

        # 数据清洗的各个步骤
        @self.app.route('/data_cleaning_step', methods=['POST'])
        @login_required
        def data_cleaning_step():
            step = request.form.get('step')
            if step == 'column_duplicate_removal':
                # 指定列去重
                action = request.form.get('action')
                columns = request.form.getlist('columns')
                if not columns:
                    flash('请选择要去重的列')
                    all_columns = self.cleaner.get_columns()
                    return render_template('data_cleaning.html', step='column_duplicate_removal',
                                           columns=all_columns)
                if action == 'remove_duplicates':
                    original_count, cleaned_count = self.cleaner.clean_duplicates(columns)
                    return render_template('data_cleaning.html', step='column_duplicate_removal',
                                           original_count=original_count, cleaned_count=cleaned_count,
                                           columns=self.cleaner.get_columns())
                return render_template('data_cleaning.html', step=step)

            elif step == 'outlier_detection':
                outlier_action = request.form.get('outlier_action')
                if not outlier_action:
                    return render_template('data_cleaning.html', step=step)
                original_count, cleaned_count = self.cleaner.handle_outliers(outlier_action)
                return render_template('data_cleaning.html', step=step, original_count=original_count,
                                       cleaned_count=cleaned_count)

            elif step == 'missing_value_info':
                missing_info = self.cleaner.get_missing_info()
                return render_template('data_cleaning.html', step=step, missing_info=missing_info)
            elif step == 'missing_value_handling':
                columns = request.form.getlist('columns')
                if not columns:
                    flash('请选择要处理的列')
                    missing_info = self.cleaner.get_missing_info()
                    return render_template('data_cleaning.html', step='missing_value_info', missing_info=missing_info)
                action = request.form.get('action')
                if action == 'delete_missing':
                    self.cleaner.delete_missing_values(columns=columns)
                elif action == 'handling_number_missing':
                    try:
                        self.cleaner.handle_numeric_missing_values(columns)
                    except ValueError as e:
                        flash(str(e))
                        missing_info = self.cleaner.get_missing_info()
                        return render_template('data_cleaning.html', step='missing_value_info',
                                               missing_info=missing_info)
                elif action == 'handling_categorical_missing':
                    try:
                        self.cleaner.handle_categorical_missing_values(columns)
                    except ValueError as e:
                        flash(str(e))
                        missing_info = self.cleaner.get_missing_info()
                        return render_template('data_cleaning.html', step='missing_value_info',
                                               missing_info=missing_info)
                missing_info = self.cleaner.get_missing_info()
                return render_template('data_cleaning.html', step='missing_value_info', missing_info=missing_info)
            elif step == 'save_cleaned_data':
                output_path = request.form.get('output_path')
                if output_path == '':
                    flash('请选择要保存的文件路径')
                    return render_template('data_cleaning.html', step=step)
                if not output_path:
                    return render_template('data_cleaning.html', step=step)
                if not os.path.exists(output_path) or os.path.isfile(output_path):
                    flash('请选择正确的文件路径')
                    return render_template('data_cleaning.html', step=step)
                self.cleaner.save_cleaned_data(output_path)
                output_path = f'{output_path}\\cleaned_data'
                if output_path not in self.cleaned_path:
                    self.cleaned_path.append(output_path)
                flash(f"文件已成功保存至 {output_path}", 'success')
                return redirect('index')

        @self.app.route('/data_integration')
        @login_required
        def data_integration():
            return render_template('data_integration.html')

        @self.app.route('/match', methods=['POST'])
        @login_required
        def match_files():
            file_left = request.files['fileLeft']
            file_right = request.files['fileRight']
            if file_left.filename.endswith('.csv'):
                file_left_name = file_left.filename.split('.csv')[0]
            elif file_left.filename.endswith('.xlsx'):
                file_left_name = file_left.filename.split('.xlsx')[0]
            else:
                raise TypeError(f'{file_left}类型不支持')
            if file_right.filename.endswith('.csv'):
                file_right_name = file_right.filename.split('.csv')[0]
            elif file_right.filename.endswith('.xlsx'):
                file_right_name = file_right.filename.split('.xlsx')[0]
            else:
                raise TypeError(f'{file_right} 类型不支持')
            self.left_cleaner = DataCleaner(file=file_left)
            self.right_cleaner = DataCleaner(file=file_right)
            self.left_cleaner.load_data()
            self.right_cleaner.load_data()
            columns_left = self.left_cleaner.get_columns()
            columns_right = self.right_cleaner.get_columns()
            # 自动匹配列（可以按列名、类型或其他逻辑来匹配）
            matched_fields = self.auto_field_matching(self.left_cleaner.data, self.right_cleaner.data, columns_left,
                                                      columns_right, file_left_name, file_right_name)
            return matched_fields

        # 数据集成中的保存逻辑
        @self.app.route('/save_file', methods=['POST'])
        @login_required
        def save_file():
            outputs = request.form.get('output_path')
            file_type = request.form.get('file_type')
            if not outputs:
                return jsonify({'success': False, 'message': '请选择保存的路径'})
            if not os.path.exists(outputs) or os.path.isfile(outputs):
                return jsonify({'success': False, 'message': '请选择正确的文件路径'})
            outputs += '\\cleaned_data'
            if outputs not in self.cleaned_path:
                self.cleaned_path.append(outputs)
            data = json.loads(request.form.get('data'))
            if file_type == '保存为xlsx文件':
                outputs_path = outputs + f"{int(time.time())}_集成.xlsx"
                self.mergeData.columns = data['columns']
                self.mergeData.to_excel(outputs_path, index=False)
            elif file_type == '保存为csv文件':
                outputs_path = outputs + f"{int(time.time())}_集成.csv"
                self.mergeData.columns = data['columns']
                self.mergeData.to_csv(outputs_path, index=False)
            else:
                return jsonify({'success': False, 'message': '文件格式不支持，请选择 .xlsx 或 .csv 格式'})
            flash(f'文件已成功保存至 {outputs}')
            return jsonify({'success': True, 'redirect_url': url_for('index')})

        @self.app.route('/feature_selection', methods=['GET', 'POST'])
        def feature_selection():
            if request.method == 'GET':
                # 获取上传文件夹中的所有文件
                cleaned_file = {}
                for path in self.cleaned_path:
                    if os.path.exists(path):
                        for file in os.listdir(path):
                            cleaned_file[f"{path}\\{file}"] = file
                return render_template('feature_selection.html', step='file_selection', files=cleaned_file)
            selected_file = request.form.get('selected_file')
            if selected_file:
                self.feature_select = FeatureSelectionAlgorithm(file_path=selected_file)
                self.feature_select.loadfile()
                all_columns = self.feature_select.get_columns()
                if len(all_columns) == len(set(all_columns)):
                    return render_template('feature_selection.html', step='select_column', columns=all_columns)
                else:
                    flash('该文件存在相同的列名，不符合')
                    return redirect(url_for('feature_selection'))
            flash('请选择一个文件')
            return redirect(url_for('feature_selection'))

        @self.app.route('/feature_selection_step', methods=['GET', 'POST'])
        def feature_selection_step():
            step = request.form.get('step')
            if step == 'enhance':
                # 进行特征选择
                response = request.form.getlist('target_column')
                # 标记增强的小数范围权重
                weight_factor = float(request.form.get('weight_factor'))
                all_columns = self.feature_select.get_columns()
                if not len(response):
                    flash('请选择目标列')
                    return render_template('feature_selection.html', step='select_column', columns=all_columns)
                self.feature_select.exchange(response)
                self.feature_select.normalization()
                self.feature_select.FeatureSelectionMarkerEnhancement(response, weight_factor)
                imgs_path = []
                for img_path in os.listdir(self.feature_select.output_dir):
                    imgs_path.append(os.path.join('static', os.path.join(
                        self.feature_select.output_dir.split('static')[-1].strip('\\'), img_path)))
                return render_template('feature_selection.html', step='enhancement_results',
                                       imgs_path=imgs_path)
            elif step == 'feature_select':
                parameter_omega = request.form.get('parameter_omega')
                if not parameter_omega:
                    return render_template('feature_selection.html', step='parameter_omega')
                else:
                    parameter_omega = float(parameter_omega)
                    result = self.feature_select.RankFeaturesBasedOnLabelDistributionUsingNeigborhoodMutualInformation(parameter_omega)
                    # feature_sort
                    feature = {}
                    for i, j in result.items():
                        feature[self.feature_select.get_columns()[i]] = j
                    self.feature_sort = dict(sorted(feature.items(), key=lambda x: x[1], reverse=True))
                    return render_template('feature_selection.html', step='feature_select',
                                           feature_sort=self.feature_sort)
            elif step == 'save_select':
                # 渲染保存路径输入页面
                return render_template('feature_selection.html', step='save_select')
            elif step == 'save_feature_result':
                output_path = request.form.get('output_path')
                if not output_path:
                    flash('请选择保存路径')
                    return render_template('feature_selection.html', step='feature_select',
                                           feature_sort=self.feature_sort)
                if not os.path.exists(output_path) or os.path.isfile(output_path):
                    flash('请选择正确的文件路劲')
                    return render_template('feature_selection.html', step='feature_select',
                                           feature_sort=self.feature_sort)
                output_path = os.path.join(output_path, 'feature_result')
                if not os.path.exists(output_path):
                    os.mkdir(output_path)
                output_path += f'\\{int(time.time())}.txt'
                try:
                    # 保存到文件
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write("Feature,Importance\n")
                        for feature, importance in self.feature_sort.items():
                            f.write(f"{feature},{importance}\n")
                    flash(f'保存成功，已保存在{output_path}')
                    return redirect(url_for('index'))
                except Exception as e:
                    flash(f'保存失败，{e}')
                    return render_template('feature_selection.html', step='feature_select',
                                           feature_sort=self.feature_sort)


