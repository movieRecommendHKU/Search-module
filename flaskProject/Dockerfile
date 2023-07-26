FROM python:3.9.17
MAINTAINER Huang
WORKDIR /app
# 工作目录，这个目录对应于镜像内的工作目录，后面的所有涉及到路径的操作都可以
# 使用WORKDIR的相对路径来指定

COPY requirements.txt requirements.txt
# 拷贝requirements.txt 到 镜像中/app/requirements.txt

RUN pip install -r requirements.txt
# 安装pip包
COPY . .
# 将当前文件中的目录复制到/app目录下

ENV FLASK_APP app
# 设置环境变量，让flask run 命令能够找到启动文件的位置

CMD ["python","-m","flask","run"]
# 执行启动命名 python -m flask run 列表中的每个元素之间代表空格
