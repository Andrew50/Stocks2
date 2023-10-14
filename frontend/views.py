from flask import Blueprint, render_template, request, redirect, url_for

admin = "dog1"
adminpswd = "dog1"

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
        ticker = request.form["ticker"]
        dt = request.form["dt"]
        timeframe = request.form["timeframe"]
        #render waiting screen
        #function
        #Return and post list onto html
        if ticker == "" or dt =="" or timeframe == "":
            return render_template("index2.html")
        else:
            return redirect(url_for("views.Return"))
    else:
        return render_template("index2.html")
    
@views.route("/return")
def Return():
    return render_template("index3.html")