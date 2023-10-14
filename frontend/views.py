from flask import Blueprint, render_template, request, redirect, url_for
import sys
import os
from Stocks2.backend.Match import Match

# # Get the absolute path of the directory containing the current script (frontend/views.py)
# current_script_dir = os.path.dirname(os.path.abspath(__file__))

# # Construct the absolute path to the backend directory
# backend_directory_path = os.path.abspath(os.path.join(current_script_dir, os.pardir, 'Stocks2', 'backend'))

# # Add the backend directory to the Python path
# sys.path.append('C:/Stocks2/backend')

# # Now you can import Match
# import Match

# Use functions or variables from Match
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
            data_out = Match.compute(data_in)
            return redirect(url_for("views.Return"))
    else:
        return render_template("index2.html")
    
@views.route("/return")
def Return():
    return render_template("index3.html", datahtml = data_out)