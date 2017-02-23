node('jenkins-slave-generic') {
    stage('Checkout') {
        checkout scm
    }

    stage('Install') {
    	sh 'virtualenv -p /usr/bin/python2.7 venv'
    	sh '. venv/bin/activate && pip install spacy sklearn scipy git+https://github.com/golastmile/rasa_nlu.git && python -m spacy.en.download all'
    }

    stage('Tests') {
    	sh '. venv/bin/activate && python -m unittest discover'
    }
}