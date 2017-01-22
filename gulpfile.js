const gulp = require('gulp')
const watch = require('gulp-watch')
const changed = require('gulp-changed')

const source = './data'
const destination = '/keybase/private/kengz,lgraesser/data';

gulp.task('default', function() {
  gulp.src([source + '/**/*', source + '/*'], { base: source })
    .pipe(watch(source, { base: source }))
    .pipe(changed(destination))
    .pipe(gulp.dest(destination));
});
