
from flask import Flask

app = Flask(__name__)

@app.route('/')
def Hello():
 return ('Im Sparta')

 app.run()
