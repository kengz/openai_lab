const _ = require('lodash')
const resolve = require('resolve-dir')

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

  function remoteCmd() {
    return (grunt.option('remote') || grunt.option('r')) ? 'xvfb-run -a -s "-screen 0 1400x900x24" --' : ''
  }

  function notiCmd(experiment) {
    return grunt.option('prod') ? `NOTI_SLACK_DEST='${config.NOTI_SLACK_DEST}' NOTI_SLACK_TOK='${config.NOTI_SLACK_TOK}' noti -k -t 'Experiment completed' -m '[${new Date().toISOString()}] ${experiment} on ${process.env.USER}'` : ''
  }

  function watchCmd() {
    return grunt.option('prod') ? 'watch' : 'shell:nowatch'
  }

  function composeCommand(experiment) {
    // override with custom command if has 'python'
    var cmd = _.includes(experiment, 'python') ? experiment : `python3 main.py -bgp -e ${experiment} -t 5`
    return `${remoteCmd()} ${cmd} | tee -a ./data/terminal.log; ${notiCmd(experiment)}`
  }

  require('load-grunt-tasks')(grunt)

  grunt.initConfig({
    sync: {
      main: {
        files: [{
          cwd: source,
          src: ['**'],
          dest: destination,
        }],
        pretend: !grunt.option('prod'), // Don't do real IO; log only
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
      nowatch: 'echo "in development; watch mode not activated"',
      finish: `echo "${finishMsg}"`,
      // TODO make smarter by autosearch
      plot: `${remoteCmd()} python3 main.py -e ${grunt.option('e')} -a`,
      clear: 'rm -rf .cache __pycache__ */__pycache__ *egg-info htmlcov .coverage data/**/ data/*.log',
    },

    concurrent: {
      default: [watchCmd(), ['lab', 'shell:finish']],
      plot: [watchCmd(), ['shell:plot', 'shell:finish']],
      options: {
        logConcurrentOutput: true
      }
    },
  })

  // grunt.event.on('watch', function(action, filepath) {
  //   // do a folder path extraction here, save to persistent file
  //   changedFiles[filepath] = action;
  //   onChange();
  // });


  grunt.registerTask('lab', 'run all the experiments', experimentTasks)
  grunt.registerTask('lab_sync', 'run lab with auto file syncing', ['concurrent:default'])
  grunt.registerTask('default', ['lab_sync'])

  grunt.registerTask('plot', ['concurrent:plot'])
  grunt.registerTask('clear', ['shell:clear'])
}
