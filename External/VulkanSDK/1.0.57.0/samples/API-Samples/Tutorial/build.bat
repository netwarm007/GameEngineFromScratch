rmdir /Q /S out
mkdir out
mkdir out\html
mkdir out\css
mkdir out\images

pushd markdown
for /R "." %%f in (*.md) do (
   perl ..\tools\Markdown.pl --html4tags %%~nf.md > ..\out\html\%%~nf.html
   )
popd
copy /y css\lg_stylesheet.css out\css\lg_stylesheet.css
copy /y images\*.png out\images
copy /y images\bg-starfield.jpg out\images\bg-starfield.jpg
