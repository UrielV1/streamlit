from multiapp import MultiApp
from apps import home, correlation_matrix, logistic_regression, xgb_model


app = MultiApp()

app.add_app("Home", home.app)
app.add_app("Correlation Matrix", correlation_matrix.app)
app.add_app("Logistic Regression model", logistic_regression.app)
app.add_app("XGB model", xgb_model.app)
#app.add_app("Multiple session comparison", two.app)

app.run()