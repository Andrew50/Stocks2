from flask import Blueprint, render_template, request, redirect, url_for
import Match

admin = "dog1"
adminpswd = "dog1"
data_in = None
views = Blueprint(__name__, "views")

@views.route("/", methods = ["POST", "GET"])
def login():
    if request.method == "POST":
        user = request.form["usr"]
        pswd = request.form["password"]
        if user == admin and pswd == adminpswd:
            return redirect(url_for("views.home"))
        else:
            return render_template("index.html")
    else:
        return render_template("index.html")

@views.route("/home", methods = ["POST", "GET"])
def home():
    if request.method == "POST":
        data_in = [request.form["ticker"],request.form["dt"],request.form["timeframe"]]
        if "" in data_in:
            return render_template("index2.html")
        else:
            data_out = match.compute(data_in)
            return redirect(url_for("views.Return"))
    else:
        return render_template("index2.html")
    
@views.route("/return")
def Return():
    return render_template("index3.html", datahtml = data_out)