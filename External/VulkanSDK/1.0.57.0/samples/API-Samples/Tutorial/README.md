# Vulkan API-Samples Tutorial README

This directory contains a tutorial "narrative" that accompanies the
samples progression source code found in the API-Samples directory
in the LunarG VulkanSamples repository.

The tutorial is authored with Markdown which is converted to HTML.
To create the HTML on a Linux platform or Windows platform with suitable
Linux emulation, run the `build.sh` bash script.
To create the HTML on a Windows platform without Linux tools, run the
`build.bat` batch file.
Both scripts creates a directory called `out` containing
the generated HTML and all the other content needed to display the tutorial
with a web browser.

Note that you must have perl installed.
The Markdown to html conversion process uses the canonical `Markdown.pl` perl script
from the creator of Markdown.
A copy of this perl script is included in this repository.

## Editing Markdown Files and Using VS Code

The "source" code for the tutorial is contained in Markdown files in the `markdown` directory.
Therefore, you should apply any changes to these files and then regenerate the HTML.

For easier Markdown editing, consider using
Microsoft's [VS Code](https://code.visualstudio.com)
editor,
which has good [Markdown support](https://code.visualstudio.com/docs/languages/markdown)
and is available on several platforms.
Several files are provided in this repository to support using this editor.

### Spell Checking with VS Code

Install the "Spelling and Grammar Checker Extension" from its [VS Code extension
website](https://marketplace.visualstudio.com/items?itemName=seanmcbreen.Spell).
The `.vscode` directory in this repository contains a spell.json file to assist
with spell checking the tutorial text.

### Markdown Lint with VS Code

Install the "Markdown Lint Extension" from its
[VS Code extension website](https://marketplace.visualstudio.com/items?itemName=DavidAnson.vscode-markdownlint).
Markdown Lint helps keep the Markdown source clean so that it converts to HTML
consistently and correctly.
Please make sure that any modifications you make to the Markdown files remain lint-free.
There is a `.markdownlint.json` file in this repository that configures Markdown lint for
this project.

## Images

The "source code" for the images are stored in XML files in the `images` directory.
These images are intended to be edited by the Google `draw.io` application.

In general, the workflow would consist of uploading the XML file into
the web application at `https://www.draw.io`,
editing the image, exporting the image to PNG, and saving the modified XML.
Then replace both the modified XML and PNG files in the repository and commit.

### Specific Steps

1. Open [draw.io](https://www.draw.io)
1. Click "Open Existing Diagram"
1. In the "Select a File" dialog, click "Upload"
1. Either drag a file into the browser or use the file selector to open the (XML) file
1. Edit the picture
1. Use the "File -> Export as -> Image..." menu item to export the PNG file. Use the Download option in the "Save As" dialog.
1. Use the "File -> Export as -> XML..." menu item to export the XML file.  Uncheck the "compressed" box.  Use the Download option in the "Save As" dialog.