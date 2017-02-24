const _ = require('lodash')
const fs = require('fs')
const resolve = require('resolve-dir')


// generic experimentId matcher. index 2: experimentId, 3 or 4: experimentName
const expIdRegex = /(\-e\s+)?(([a-zA-Z0-9_]+)\-\d{4}_\d{2}_\d{2}_\d{6}|([a-zA-Z0-9_]+))/
const historyPath = './config/history.json'
const finishMsg = `
===========================================
Experiments complete. Press Ctrl+C to exit.
===========================================
`


module.exports = function(grunt) {
  process.env.NODE_ENV = grunt.option('prod') ? 'production' : 'development'

  const config = require('config')
  const dataSrc = 'data'
  const dataDest = resolve(config.data_sync_destination)
  const experiments = config.experiments
  const experimentTasks = _.map(experiments, function(name) {
    return `shell:experiment:${name}`
  })

  function writeHistory(history) {
    fs.writeFileSync(historyPath, JSON.stringify(history, null, 2))
    return history
  }

  function readHistory() {
    if (grunt.option('resume')) {
      try {
        return JSON.parse(fs.readFileSync(historyPath, 'utf8'))
      } catch (err) {
        grunt.log.ok(`No existing ${historyPath} to resume, creating new`)
        return writeHistory({})
      }
    } else {
      return {}
    }
  }

  let history = readHistory()

  function updateHistory(filepath) {
    if (fs.lstatSync(filepath).isFile()) {
      // only interested in data folder, skip otherwise
      return
    }
    const matchedPath = filepath.split('/').pop().match(expIdRegex)
    if (matchedPath) {
      const experimentId = matchedPath[2]
      const experimentName = matchedPath[3] || matchedPath[4]
      history[experimentName] = experimentId
      writeHistory(history)
    }
  }

  function remoteCmd() {
    return (grunt.option('remote')) ? 'xvfb-run -a -s "-screen 0 1400x900x24" --' : ''
  }

  function plotCmd() {
    return grunt.option('plotOnly') ? ' -a' : ''
  }

  function notiCmd(experiment) {
    return (grunt.option('prod') && !grunt.option('plotOnly')) ? `NOTI_SLACK_DEST='${config.NOTI_SLACK_DEST}' NOTI_SLACK_TOK='${config.NOTI_SLACK_TOK}' noti -k -t 'Experiment completed' -m '[${new Date().toISOString()}] ${experiment} on ${process.env.USER}'` : ''
  }

  function resumeExperimentStr(eStr) {
    const matchedExp = eStr.match(expIdRegex)
    if (matchedExp) {
      const experimentIdOrName = matchedExp[2]
      const experimentName = matchedExp[3] || matchedExp[4]
      if (history[experimentName]) {
        return eStr.replace(experimentIdOrName, history[experimentName])
      }
    }
    return eStr
  }

  function composeCommand(experimentStr) {
    var eStr = experimentStr
    if (grunt.option('resume')) {
      eStr = resumeExperimentStr(eStr)
    }

    // override with custom command if has 'python'
    var pyCmd = _.includes(eStr, 'python') ? eStr : `python3 main.py -bgp -e ${eStr} -t 5${plotCmd()}`
    const cmd = `${remoteCmd()} ${pyCmd} | tee -a ./data/terminal.log; ${notiCmd(eStr)}`
    grunt.log.ok(`Composed command: ${cmd}`)
    return cmd
  }


  require('load-grunt-tasks')(grunt)

  grunt.initConfig({
    sync: {
      main: {
        files: [{
          cwd: dataSrc,
          src: ['**'],
          dest: dataDest,
        }],
        pretend: !grunt.option('prod'), // Don't do real IO
      }
    },

    watch: {
      data: {
        files: `${dataSrc}/**`,
        tasks: ['sync'],
        options: {
          debounceDelay: 60000,
          interval: 60000,
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
      experiment: {
        command(experimentStr) {
          return composeCommand(experimentStr)
        },
        options: {
          stdout: true
        }
      },
      finish: `echo "${finishMsg}"`,
      clear: 'rm -rf .cache __pycache__ */__pycache__ *egg-info htmlcov .coverage data/**/ data/*.log config/history.json',
    },

    concurrent: {
      default: ['watch', ['lab', 'shell:finish']],
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

  grunt.registerTask('plot', function() {
    grunt.option('plotOnly', true)
    grunt.option('resume', true)
    grunt.task.run('default')
  })
  grunt.registerTask('clear', ['shell:clear'])
}
