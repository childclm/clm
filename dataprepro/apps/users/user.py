import flask_login


class User(flask_login.UserMixin):
    def __init__(self, user_id):
        """
        初始化用户对象
        :param user_id: 用户的唯一标识符
        """
        self.id = user_id  # Flask-Login 使用 'id' 属性作为用户标识

    def get_id(self):
        """
        返回用户的唯一标识符，供 Flask-Login 使用。
        """
        return str(self.id)
