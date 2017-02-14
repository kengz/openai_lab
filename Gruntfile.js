const _ = require('lodash')
const config = require('config')
const resolve = require('resolve-dir')

const source = './data'
const destination = resolve(config.data_sync_destination)
const experiments = config.experiments
const experimentTasks = _.map(experiments, function(name) {
  return `shell:local:${name}`
})


const finishMsg = `
===========================================
Experiments complete. Press Ctrl+C to exit.
===========================================
`

module.exports = function(grunt) {
  require('load-grunt-tasks')(grunt)

  function remoteCmd() {
    return (grunt.option('remote') || grunt.option('r')) ? 'xvfb-run -a -s "-screen 0 1400x900x24" --' : ''
  }

  function composeCommand(experiment) {
    // override with custom command if has 'python'
    var cmd = _.includes(experiment, 'python') ? experiment : `python3 main.py -bgp -e ${experiment} -t 5`
    return `${remoteCmd()} ${cmd} | tee -a ./data/terminal.log; NOTI_SLACK_DEST='${config.NOTI_SLACK_DEST}' NOTI_SLACK_TOK='${config.NOTI_SLACK_TOK}' noti -k -t 'Experiment completed' -m '[${new Date().toISOString()}] ${experiment} on ${process.env.USER}'`
  }

  grunt.initConfig({
    sync: {
      main: {
        files: [{
          cwd: source,
          src: ['**'],
          dest: destination,
        }],
        // pretend: true, // Don't do real IO; log only
        // verbose: true // Display log messages when copying files
      }
    },

    watch: {
      data: {
        files: `${source}/**`,
        tasks: ['sync']
      }
    },

    shell: {
      options: {
        execOptions: {
          killSignal: 'SIGINT',
          env: process.env
        }
      },
      local: {
        command(experiment) {
          return composeCommand(experiment)
        }
      },
      finish: `echo "${finishMsg}"`,
      // TODO make smarter by autosearch
      plot: `${remoteCmd()} python3 main.py -e ${grunt.option('e')} -a`,
      // TODO add a dev mode clear vs prod mode clear
      clear: `rm -rf .cache __pycache__ */__pycache__ *egg-info htmlcov .coverage data/**/ data/*.log`,
    },

    concurrent: {
      local: ['watch', ['lab', 'shell:finish']],
      plot: ['watch', ['shell:plot', 'shell:finish']],
      options: {
        logConcurrentOutput: true
      }
    },
  })

  grunt.registerTask('lab', 'run all the experiments', experimentTasks)
  grunt.registerTask('lab_sync', 'run lab with auto file syncing', ['concurrent:local'])
  grunt.registerTask('default', ['lab_sync'])

  grunt.registerTask('plot', ['concurrent:plot'])
  grunt.registerTask('clear', ['shell:clear'])
}
