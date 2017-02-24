node('jenkins-slave-tensorflow') {
    stage('Checkout') {
        checkout scm
    }

    stage('Setup') {
        sh 'python setup.py install'
    }

    stage('Tests') {
        sh 'python -m unittest discover'
    }
}