def packagePath = "snips_nlu"
def VENV = ". venv/bin/activate"


def version(path) {
    readFile("${path}/__version__").split("\n")[0]
}

def executeInVirtualEnv(pythonPath, venvPath, cmd) {
    def credentials = "${env.NEXUS_USERNAME_PYPI}:${env.NEXUS_PASSWORD_PYPI}"
    sh """
    rm -rf $venvPath
    virtualenv -p $pythonPath $venvPath
    . ${venvPath}/bin/activate
    echo "[global]\nindex = https://${credentials}@nexus-repository.snips.ai/repository/pypi-internal/pypi\nindex-url = https://pypi.python.org/simple/\nextra-index-url = https://${credentials}@nexus-repository.snips.ai/repository/pypi-internal/simple" >> ${venvPath}/pip.conf
    $cmd
    """
}

def uploadWheel(pythonPath, venvPath) {
    cmd = "python setup.py bdist_wheel upload -r pypisnips"
    executeInVirtualEnv(pythonPath, venvPath, cmd)
}

def buildWheel(pythonPath, venvPath) {
    cmd = "python setup.py bdist_wheel"
    executeInVirtualEnv(pythonPath, venvPath, cmd)
}

def setupAndBuild(pythonPath, venvPath) {
    stage('Checkout') {
        deleteDir()
        checkout scm
        sh "git submodule update --init --recursive"
    }

    stage('Build') {
        executeInVirtualEnv(pythonPath, venvPath, "pip install .[test]")
    }

    stage('Tests') {
        def branchName = "${env.BRANCH_NAME}"

        sh """
        . ${venvPath}/bin/activate
        python -m unittest discover
        """

        if(branchName.startsWith("release/") || branchName.startsWith("hotfix/") || branchName == "master") {
            sh "python -m unittest discover -p 'integration_test*.py'"
        }
    }
}

node('macos') {
    def branchName = "${env.BRANCH_NAME}"

    def python27path = sh(returnStdout: true, script: 'which python2.7').trim()
    def python34path = sh(returnStdout: true, script: 'which python3.4').trim()
    def python35path = sh(returnStdout: true, script: 'which python3.5').trim()
    def python36path = sh(returnStdout: true, script: 'which python3.6').trim()

    setupAndBuild(python27path, "venv27")
    setupAndBuild(python34path, "venv34")
    setupAndBuild(python35path, "venv35")
    setupAndBuild(python36path, "venv36")

    switch (branchName) {
        case "master":
            stage('Publish') {
                deleteDir()
                checkout scm
                def rootPath = pwd()
                def path = "${rootPath}/${packagePath}"
                def newVersion = version(path)

                def gitTagExists = sh(returnStdout: true, script: 'git tag -l $newVersion')

                if (gitTagExists) {
                    echo "tag $newVersion already exists."
                    exit 1
                }

                sh """
                git tag $newVersion
                git remote rm origin
                git remote add origin 'git@github.com:snipsco/snips-nlu.git'
                git config --global user.email 'jenkins@snips.ai'
                git config --global user.name 'Jenkins'
                git push --tags
                """
            }

            stage('Upload wheels') {
                uploadWheel(python27path, venv27path)
                uploadWheel(python34path, venv34path)
                uploadWheel(python35path, venv35path)
                uploadWheel(python36path, venv36path)
            }
        default:
            stage('Build wheels') {
                buildWheel(python27path, venv27path)
                buildWheel(python34path, venv34path)
                buildWheel(python35path, venv35path)
                buildWheel(python36path, venv36path)
            }
    }
}
