def branchName = "${env.BRANCH_NAME}"
def VENV = ". venv/bin/activate"

node('jenkins-slave-generic') {
    stage('Checkout') {
	    deleteDir()
        checkout scm
        sh "git submodule update --init --recursive"
    }

    stage('Setup') {
        def credentials = "${env.NEXUS_USERNAME_PYPI}:${env.NEXUS_PASSWORD_PYPI}"
    	sh "virtualenv venv"
    	sh """
    	${VENV}
    	echo "[global]\nindex = https://${credentials}@nexus-repository.snips.ai/repository/pypi-internal/pypi\nindex-url = https://pypi.python.org/simple/\nextra-index-url = https://${credentials}@nexus-repository.snips.ai/repository/pypi-internal/simple" >> venv/pip.conf
    	pip install .
    	"""
    }

    stage('Tests') {
        sh """
        ${VENV}
        python -m unittest discover
        """
    }

    stage('Publish') {
        def credentials = "${env.NEXUS_USERNAME_PYPI}:${env.NEXUS_PASSWORD_PYPI}"
        switch (branchName) {
            case "master":
                sh """
                . venv/bin/activate
                python setup.py bdist_wheel upload -r https://${credentials}@nexus-repository.snips.ai/repository/pypi-internal/
                """
            default:
                sh """
                . venv/bin/activate
                python setup.py bdist_wheel
                """
        }
    }
}