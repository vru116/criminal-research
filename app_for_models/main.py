import flask
import joblib

models = {
          "gender_accused" : {
           "file" : 'catboost_model_gender_.joblib',
           "model" : None
           },

          "prior_convictions" : {
           "file" : 'crime_motive_model.joblib',
           "model" : None
           },
           
          "location" : {
           "file" : 'crime_motive_model.joblib',
           "model" : None
           },

          "gender_victim" : {
           "file" : 'crime_motive_model.joblib',
           "model" : None
           },

          "prison_term" : {
           "file" : 'crime_motive_model.joblib',
           "model" : None
           },

          "alcohol" : {
           "file" : 'crime_motive_model.joblib',
           "model" : None
           },

          "motive" : {
           "file" : 'crime_motive_model.joblib',
           "model" : None
           },

          "method" : {
           "file" : 'crime_motive_model.joblib',
           "model" : None
           },

          "prior_convictions" : {
           "file" : 'crime_motive_model1.joblib',
           "model" : None
           },
        }
          

app = flask.Flask(__name__)

@app.route("/")
@app.route("/index.html")
def html_index():
    return flask.render_template('index.html')

@app.route("/model.html")
def html_model():
    return flask.render_template('model.html')

@app.route("/result.html", methods=['GET'])
def html_result():
    mname = flask.request.args.get('model')
    if mname and (mname != '') and (mname in models) and (models[mname]["model"] is not None):
        x = []
        model = models[mname]["model"]
        for f in model["features"]:
            val = flask.request.args.get(f)
            if (val):
                x.append(val)
            else:
                x.append('')
        print(x)
        y = model["model"].predict(x)
        result = '';
        if hasattr(y, "__len__"):
            result = y[0]
        else:
            result = y;
        return result
    else:
        return "ОШИБКА"


if __name__ == '__main__':
    for key, value in models.items():
        print("Loading model ", key, "...", end="")
        try:
            value["model"] = joblib.load("models/"+value["file"])
        except:
            value["model"] = None
        if (value["model"]):
            print("OK!")
        else:
            print("Error!")
        
    app.run()