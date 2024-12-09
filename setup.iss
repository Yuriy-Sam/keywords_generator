[Setup]
AppName=KeywordCraze
AppVersion=1.1
DefaultDirName={pf}\KeywordCraze
DefaultGroupName=KeywordCraze
OutputDir=.\output
OutputBaseFilename=KeywordCraze
DisableDirPage=yes

[Files]
Source: "dist/KeywordCraze.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: ".env"; DestDir: "{app}"; Flags: ignoreversion
Source: "assets\*.*"; DestDir: "{app}\assets"; Flags: ignoreversion recursesubdirs
Source: "update_installer.exe"; DestDir: "{app}";
Source: "version.txt"; DestDir: "{app}"; Flags: ignoreversion

[Tasks]
Name: desktopicon; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:";
Name: "startmenu"; Description: "Create a &Start Menu shortcut"; GroupDescription: "Additional icons:";

[Icons]
Name: "{commondesktop}\Keyword Craze"; Filename: "{app}\KeywordCraze.exe"; WorkingDir: "{app}"; Tasks: desktopicon
Name: "{group}\KeywordCraze"; Filename: "{app}\KeywordCraze.exe"; WorkingDir: "{app}"; Tasks: startmenu
Name: "{group}\Uninstall KeywordCraze"; Filename: "{uninstallexe}"; Tasks: startmenu

[Run]
Filename: "{app}\KeywordCraze.exe"; Description: "Run KeywordCraze"; Flags: nowait postinstall skipifsilent


[CustomMessages]
en=KeywordCraze - Setup Complete!; After installation is complete, you can launch KeywordCraze from your desktop or Start Menu.

[UninstallDelete]
Type: files; Name: "{app}\KeywordCraze.exe";