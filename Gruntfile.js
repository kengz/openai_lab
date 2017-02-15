const _ = require('lodash')
const fs = require('fs')
const resolve = require('resolve-dir')


// generic experimentId matcher. index 2: experimentId, 3 or 4: experimentName
const expIdRegex = /(\-e\s+)?(([a-zA-Z0-9_]+)_\d{4}\-\d{2}\-\d{2}_\d{6}|([a-zA-Z0-9_]+))/
const historyPath = './config/history.json'
const finishMsg = `
===========================================
Experiments complete. Press Ctrl+C to exit.
===========================================
`


module.exports = function(grunt) {
  process.env.NODE_ENV = grunt.option('prod') ? 'production' : 'development'
  const config = require('config')
  const source = './data'
  const destination = resolve(config.data_sync_destination)
  const experiments = config.experiments
  const experimentTasks = _.map(experiments, function(name) {
    return `shell:exp:${name}`
  })

  function readHistory() {
    if (grunt.option('resume')) {
      try {
        return JSON.parse(fs.readFileSync(historyPath, 'utf8'))
      } catch (err) {
        console.log(`No existing ${historyPath} to resume, creating new`)
        newHistory = {}
        writeHistory(newHistory)
        return newHistory
      }
    } else {
      return {}
    }
  }

  let history = readHistory()

  function writeHistory(history) {
    fs.writeFile(historyPath, JSON.stringify(history, null, 2))
  }

  function updateHistory(filepath) {
    const matchedPath = filepath.match(expIdRegex)
    if (matchedPath) {
      const experimentId = matchedPath[2]
      const experimentName = matchedPath[3] || matchedPath[4]
      history[experimentName] = experimentId
      writeHistory(history)
    }
  }

  function remoteCmd() {
    return (grunt.option('remote') || grunt.option('r')) ? 'xvfb-run -a -s "-screen 0 1400x900x24" --' : ''
  }

  function notiCmd(experiment) {
    return grunt.option('prod') ? `NOTI_SLACK_DEST='${config.NOTI_SLACK_DEST}' NOTI_SLACK_TOK='${config.NOTI_SLACK_TOK}' noti -k -t 'Experiment completed' -m '[${new Date().toISOString()}] ${experiment} on ${process.env.USER}'` : ''
  }

  function composeCommand(experiment) {
    if (grunt.option('resume')) {
      // search and replace using history
      // but if preinit hmm history is empty, so just dont replace
    }
    // override with custom command if has 'python'
    var cmd = _.includes(experiment, 'python') ? experiment : `python3 main.py -bgp -e ${experiment} -t 5`
    return `${remoteCmd()} ${cmd} | tee -a ./data/terminal.log; ${notiCmd(experiment)}`
  }



  require('load-grunt-tasks')(grunt)

  grunt.initConfig({
    sync: {
      main: {
        files: [{
          src: [`${source}/**`],
          dest: destination,
        }],
        pretend: !grunt.option('prod'), // Don't do real IO
      }
    },

    watch: {
      data: {
        files: `${source}/**`,
        tasks: ['sync'],
        options: {
          debounceDelay: 60000,
        },
      }
    },

    shell: {
      options: {
        execOptions: {
          killSignal: 'SIGINT',
          env: process.env
        }
      },
      exp: {
        command(experiment) {
          return composeCommand(experiment)
        }
      },
      finish: `echo "${finishMsg}"`,
      // TODO make smarter by autosearch
      plot: `${remoteCmd()} python3 main.py -e ${grunt.option('e')} -a`,
      clear: 'rm -rf .cache __pycache__ */__pycache__ *egg-info htmlcov .coverage data/**/ data/*.log',
    },

    concurrent: {
      default: ['watch', ['lab', 'shell:finish']],
      plot: ['watch', ['shell:plot', 'shell:finish']],
      options: {
        logConcurrentOutput: true
      }
    },
  })

  grunt.event.on('watch', function(action, filepath) {
    updateHistory(filepath)
  })


  grunt.registerTask('lab', 'run all the experiments', experimentTasks)
  grunt.registerTask('lab_sync', 'run lab with auto file syncing', ['concurrent:default'])
  grunt.registerTask('default', ['lab_sync'])

  grunt.registerTask('plot', ['concurrent:plot'])
  grunt.registerTask('clear', ['shell:clear'])
}
