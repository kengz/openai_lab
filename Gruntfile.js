const _ = require('lodash')
const config = require('config')
const resolve = require('resolve-dir')

const source = './data'
const destination = resolve(config.data_sync_destination)
const experiments = config.experiments
const experimentTasks = _.map(experiments, function(name) {
  return `shell:run:${name}`
})


module.exports = function(grunt) {
  require('load-grunt-tasks')(grunt)

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
      run: {
        command (experiment) {
          var cmd = `python3 main.py -bgp -e ${experiment} -t 5`
          if (_.includes(experiment, 'python')) {
            // override with custom command
            cmd = experiment
          }
          return `(${cmd} | tee -a ./data/terminal.log) & NOTI_SLACK_DEST='${config.NOTI_SLACK_DEST}' NOTI_SLACK_TOK='${config.NOTI_SLACK_TOK}' noti -k -t '${experiment}' -pwatch $! &`
        }
      },
    },

    concurrent: {
      target: ['watch', 'lab'],
      options: {
        logConcurrentOutput: true
      }
    },
  })

  grunt.registerTask('lab', 'run all the experiments', experimentTasks)
  grunt.registerTask('lab_sync', 'run lab with auto file syncing', ['concurrent'])
  grunt.registerTask('default', ['lab_sync'])
}
