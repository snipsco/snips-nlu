def packagePath = "snips_nlu"


def version(path) {
    readFile("${path}/__version__").split("\n")[0]
}

def checkout() {
    deleteDir()
    checkout scm
    sh "git submodule update --init --recursive"
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

def uploadAssets() {
    def pythonPath = sh(returnStdout: true, script: 'which python2.7').trim()
    cmd_wheel = "python setup.py bdist_wheel --universal upload -r pypisnips"
    cmd_sdist = "python setup.py sdist upload -r pypisnips"
    executeInVirtualEnv(pythonPath, "venv", cmd_wheel)
    executeInVirtualEnv(pythonPath, "venv", cmd_sdist)
}

def buildWheel() {
    def pythonPath = sh(returnStdout: true, script: 'which python2.7').trim()
    cmd = "python setup.py bdist_wheel --universal"
    executeInVirtualEnv(pythonPath, "venv", cmd)
}

def installAndTest(pythonPath, venvPath, includeIntegrationTest=false, includeLintingTest=false, includeSampleTest=false) {
    stage('Checkout') {
        checkout()
    }

    stage('Build') {
        def branchName = "${env.BRANCH_NAME}"
        if(includeIntegrationTest && (branchName.startsWith("release/") || branchName.startsWith("hotfix/") || branchName == "master")) {
            executeInVirtualEnv(pythonPath, venvPath, "pip install '.[test, integration_test]'")
        } else {
            executeInVirtualEnv(pythonPath, venvPath, "pip install .[test]")
        }
    }

    stage('Tests') {
        def branchName = "${env.BRANCH_NAME}"

        sh """
        . ${venvPath}/bin/activate
        python -m unittest discover
        """

        if(includeLintingTest) {
            sh """
            . ${venvPath}/bin/activate
            python -m unittest discover -p 'linting_test*.py'
            """
        }

        if(includeSampleTest) {
            sh """
            . ${venvPath}/bin/activate
            python -m unittest discover -p 'samples_test*.py'
            """
        }

        if(includeIntegrationTest) {
            sh """
            . ${venvPath}/bin/activate
            python -m unittest discover -p 'integration_test*.py'
            """
        }
    }
}

node('macos') {
    def branchName = "${env.BRANCH_NAME}"

    def python27path = sh(returnStdout: true, script: 'which python2.7').trim()
    def python34path = sh(returnStdout: true, script: 'which python3.4').trim()
    def python35path = sh(returnStdout: true, script: 'which python3.5').trim()
    def python36path = sh(returnStdout: true, script: 'which python3.6').trim()

    if(branchName.startsWith("release/") || branchName.startsWith("hotfix/") || branchName == "master") {
        installAndTest(python36path, "venv36", true, true, true)
        installAndTest(python27path, "venv27")
        installAndTest(python34path, "venv34")
        installAndTest(python35path, "venv35")
    } else {
        installAndTest(python36path, "venv36", true, true)
        installAndTest(python27path, "venv27")
    }

    if (branchName == "master") {
        stage('Publish') {
            checkout()
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

        stage('Upload assets') {
            uploadAssets()
        }
    } else {
        stage('Build wheel') {
            buildWheel()
        }
    }
}
