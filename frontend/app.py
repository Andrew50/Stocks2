from flask import Flask
from views import views

app = Flask(__name__)
app.register_blueprint( views, url_prefix  = "/view")



@app.route("/")
def home():
    return "this is the homne page"

if __name__== '__main__':
    app.run(debug=True)