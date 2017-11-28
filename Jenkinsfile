def branchName = "${env.BRANCH_NAME}"
def packagePath = "snips_nlu"
def VENV = ". venv/bin/activate"


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
    	def credentials = "${env.NEXUS_USERNAME_PYPI}:${env.NEXUS_PASSWORD_PYPI}"
    	sh """
    	${VENV}
    	echo "[global]\nindex = https://${credentials}@nexus-repository.snips.ai/repository/pypi-internal/pypi\nindex-url = https://pypi.python.org/simple/\nextra-index-url = https://${credentials}@nexus-repository.snips.ai/repository/pypi-internal/simple" >> venv/pip.conf
    	pip install .[test]
    	"""
    }

    stage('Tests') {
        if(branchName.startsWith("release/") || branchName.startsWith("hotfix/") || branchName == "master") {
            sh """
            ${VENV}
            python -m unittest discover
            python -m unittest discover -p 'integration_test*.py'
            """
        } else {
            sh """
            ${VENV}
            python -m unittest discover
            """
        }
    }

    stage('Publish') {
        switch (branchName) {
            case "master":
                deleteDir()
                checkout scm
                def rootPath = pwd()
                def path = "${rootPath}/${packagePath}"
                def credentials = "${env.NEXUS_USERNAME_PYPI}:${env.NEXUS_PASSWORD_PYPI}"
                sh "git submodule update --init --recursive"
                sh """
                virtualenv venv
                ${VENV}
                echo "[global]\nindex = https://${credentials}@nexus-repository.snips.ai/repository/pypi-internal/pypi\nindex-url = https://pypi.python.org/simple/\nextra-index-url = https://${credentials}@nexus-repository.snips.ai/repository/pypi-internal/simple" >> venv/pip.conf
                pip install .
                git tag ${version(path)}
                git remote rm origin
                git remote add origin 'git@github.com:snipsco/snips-nlu.git'
                git config --global user.email 'jenkins@snips.ai'
                git config --global user.name 'Jenkins'
                git push --tags
                python setup.py bdist_wheel upload -r pypisnips
                """
            default:
                sh """
                ${VENV}
                python setup.py bdist_wheel
                """
        }
    }
}
