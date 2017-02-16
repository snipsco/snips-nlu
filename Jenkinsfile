node('jenkins-slave-generic') {
    stage('Checkout') {
        checkout scm
    }

    stage('Tests') {
        sh 'python -m unittest discover'
    }
}