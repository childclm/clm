from flask_login import login_user
from flask_wtf import FlaskForm
from dataprepro.apps.users.user import User
from dataprepro.apps.utils.save_captcha import RedisClient
from flask import flash, url_for, request, redirect, render_template
from typing import Optional
import random


class Login:
    def __init__(self):
        self.db: Optional[RedisClient] = None
        self.unique_id = None

    def random_color(self):
        """生成随机颜色"""
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def handle_login(self, form):
        """处理登录逻辑"""
        username = form.username.data
        password = form.password.data
        captcha = form.captcha.data
        # 验证逻辑
        self.db = RedisClient()
        saved_captcha = self.db.get(self.unique_id)
        if not saved_captcha:
            flash('验证码已过期，请重新获取', 'error')
            return render_template('login.html', form=form)

        if captcha != saved_captcha:
            flash('验证码错误，请重试', 'error')
            return render_template('login.html', form=form)

        if username == 'admin' and password == 'admin':
            user = User(user_id=username)
            login_user(user)  # 登录用户
            flash('登录成功！', 'success')
            return redirect(url_for('index'))

        flash('账号或密码不正确，请重试', 'error')
        return render_template('login.html', form=form)


