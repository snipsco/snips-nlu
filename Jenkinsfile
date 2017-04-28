def branchName = "${env.BRANCH_NAME}"
def packagePath = "snips_nlu"
def VENV = ". venv/bin/activate"
def credentials = "${env.NEXUS_USERNAME_PYPI}:${env.NEXUS_PASSWORD_PYPI}"

def version(path) {
    readFile("${path}/__version__").split("\n")[0]
}


node('jenkins-slave-generic') {
    stage('Checkout') {
        deleteDir()
        checkout scm
        sh "git submodule update --init --recursive"
    }

    stage('Setup') {
    	sh "virtualenv venv"
    	sh """
    	${VENV}
    	echo "[global]\nindex = https://${credentials}@nexus-repository.snips.ai/repository/pypi-internal/pypi\nindex-url = https://pypi.python.org/simple/\nextra-index-url = https://${credentials}@nexus-repository.snips.ai/repository/pypi-internal/simple" >> venv/pip.conf
    	pip install .[test]
    	"""
    }

    stage('Tests') {
        sh """
        ${VENV}
        python -m unittest discover
        """
    }

    stage('Publish') {
        switch (branchName) {
            case "master":
                deleteDir()
                checkout scm
                def rootPath = pwd()
                def path = "${rootPath}/${packagePath}"
                sh "git submodule update --init --recursive"
                sh """
                virtualenv venv
                ${VENV}
                echo "[global]\nindex = https://${credentials}@nexus-repository.snips.ai/repository/pypi-internal/pypi\nindex-url = https://pypi.python.org/simple/\nextra-index-url = https://${credentials}@nexus-repository.snips.ai/repository/pypi-internal/simple" >> venv/pip.conf
                pip install .
                python setup.py bdist_wheel upload -r pypisnips
                git tag ${version(path)}
                git push --tags
                """
            default:
                sh """
                ${VENV}
                python setup.py bdist_wheel
                """
        }
    }
}
