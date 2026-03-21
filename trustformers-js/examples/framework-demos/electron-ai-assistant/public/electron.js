const { app, BrowserWindow, Menu, ipcMain, dialog, shell, nativeTheme } = require('electron');
const { autoUpdater } = require('electron-updater');
const Store = require('electron-store');
const path = require('path');
const isDev = require('electron-is-dev');

// Enable live reload for Electron too
if (isDev) {
  require('electron-reload')(__dirname, {
    electron: path.join(__dirname, '..', 'node_modules', '.bin', 'electron'),
    hardResetMethod: 'exit'
  });
}

// Configure auto-updater
autoUpdater.logger = require('electron-log');
autoUpdater.logger.transports.file.level = 'info';

// Initialize persistent store
const store = new Store({
  defaults: {
    windowBounds: { width: 1200, height: 800 },
    theme: 'system',
    aiSettings: {
      model: 'default',
      temperature: 0.7,
      maxTokens: 150,
      autoSave: true,
    },
    conversations: [],
    preferences: {
      autoStart: false,
      minimizeToTray: true,
      notifications: true,
      saveHistory: true,
    }
  }
});

let mainWindow;
let splashWindow;

// Security: Configure CSP and other security headers
const securityHeaders = {
  'Content-Security-Policy': [
    "default-src 'self'",
    "script-src 'self' 'unsafe-inline'",
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: https:",
    "font-src 'self' data:",
    "connect-src 'self' https:",
  ].join('; ')
};

function createSplashWindow() {
  splashWindow = new BrowserWindow({
    width: 400,
    height: 300,
    frame: false,
    alwaysOnTop: true,
    transparent: true,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    }
  });

  splashWindow.loadFile(path.join(__dirname, 'splash.html'));
  
  splashWindow.on('closed', () => {
    splashWindow = null;
  });
}

function createWindow() {
  // Get stored window bounds
  const { width, height } = store.get('windowBounds');

  // Create the browser window
  mainWindow = new BrowserWindow({
    width,
    height,
    minWidth: 800,
    minHeight: 600,
    icon: path.join(__dirname, '../assets/icon.png'),
    show: false, // Don't show until ready-to-show
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      enableRemoteModule: false,
      preload: path.join(__dirname, 'preload.js'),
      webSecurity: !isDev, // Disable only in development
      allowRunningInsecureContent: false,
      experimentalFeatures: false,
    }
  });

  // Set security headers
  mainWindow.webContents.session.webRequest.onHeadersReceived((details, callback) => {
    callback({
      responseHeaders: {
        ...details.responseHeaders,
        ...securityHeaders
      }
    });
  });

  // Load the app
  const startUrl = isDev 
    ? 'http://localhost:3000' 
    : `file://${path.join(__dirname, '../build/index.html')}`;
  
  mainWindow.loadURL(startUrl);

  // Show window when ready
  mainWindow.once('ready-to-show', () => {
    if (splashWindow) {
      splashWindow.close();
    }
    
    mainWindow.show();
    
    // Focus on window
    if (isDev) {
      mainWindow.webContents.openDevTools();
    }
  });

  // Handle window closed
  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  // Save window bounds on resize/move
  mainWindow.on('resize', () => {
    store.set('windowBounds', mainWindow.getBounds());
  });

  mainWindow.on('move', () => {
    store.set('windowBounds', mainWindow.getBounds());
  });

  // Handle external links
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });

  // Prevent navigation away from the app
  mainWindow.webContents.on('will-navigate', (event, navigationUrl) => {
    const parsedUrl = new URL(navigationUrl);
    
    if (parsedUrl.origin !== 'http://localhost:3000' && parsedUrl.origin !== startUrl) {
      event.preventDefault();
    }
  });
}

// Create application menu
function createMenu() {
  const template = [
    {
      label: 'File',
      submenu: [
        {
          label: 'New Conversation',
          accelerator: 'CmdOrCtrl+N',
          click: () => {
            mainWindow.webContents.send('menu:new-conversation');
          }
        },
        {
          label: 'Save Conversation',
          accelerator: 'CmdOrCtrl+S',
          click: async () => {
            const result = await dialog.showSaveDialog(mainWindow, {
              defaultPath: 'conversation.json',
              filters: [
                { name: 'JSON Files', extensions: ['json'] },
                { name: 'Text Files', extensions: ['txt'] }
              ]
            });
            
            if (!result.canceled) {
              mainWindow.webContents.send('menu:save-conversation', result.filePath);
            }
          }
        },
        { type: 'separator' },
        {
          label: 'Preferences',
          accelerator: 'CmdOrCtrl+,',
          click: () => {
            mainWindow.webContents.send('menu:open-preferences');
          }
        },
        { type: 'separator' },
        { role: 'quit' }
      ]
    },
    {
      label: 'Edit',
      submenu: [
        { role: 'undo' },
        { role: 'redo' },
        { type: 'separator' },
        { role: 'cut' },
        { role: 'copy' },
        { role: 'paste' },
        { role: 'selectall' }
      ]
    },
    {
      label: 'View',
      submenu: [
        { role: 'reload' },
        { role: 'forceReload' },
        { role: 'toggleDevTools' },
        { type: 'separator' },
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' },
        { type: 'separator' },
        { role: 'togglefullscreen' },
        {
          label: 'Toggle Theme',
          accelerator: 'CmdOrCtrl+Shift+T',
          click: () => {
            const currentTheme = store.get('theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            store.set('theme', newTheme);
            mainWindow.webContents.send('theme:changed', newTheme);
          }
        }
      ]
    },
    {
      label: 'AI',
      submenu: [
        {
          label: 'Clear History',
          click: () => {
            dialog.showMessageBox(mainWindow, {
              type: 'warning',
              buttons: ['Cancel', 'Clear'],
              defaultId: 0,
              message: 'Clear all conversation history?',
              detail: 'This action cannot be undone.'
            }).then(result => {
              if (result.response === 1) {
                store.set('conversations', []);
                mainWindow.webContents.send('menu:clear-history');
              }
            });
          }
        },
        {
          label: 'Export Data',
          click: async () => {
            const result = await dialog.showSaveDialog(mainWindow, {
              defaultPath: 'trustformers-data.json',
              filters: [{ name: 'JSON Files', extensions: ['json'] }]
            });
            
            if (!result.canceled) {
              const data = {
                conversations: store.get('conversations'),
                settings: store.get('aiSettings'),
                preferences: store.get('preferences'),
                exportDate: new Date().toISOString()
              };
              
              require('fs').writeFileSync(result.filePath, JSON.stringify(data, null, 2));
            }
          }
        }
      ]
    },
    {
      label: 'Window',
      submenu: [
        { role: 'minimize' },
        { role: 'close' }
      ]
    },
    {
      label: 'Help',
      submenu: [
        {
          label: 'About TrustformeRS AI Assistant',
          click: () => {
            dialog.showMessageBox(mainWindow, {
              type: 'info',
              title: 'About TrustformeRS AI Assistant',
              message: 'TrustformeRS AI Assistant',
              detail: `Version: ${app.getVersion()}\nElectron: ${process.versions.electron}\nNode: ${process.versions.node}\n\nA cross-platform desktop AI assistant powered by TrustformeRS machine learning capabilities.`
            });
          }
        },
        {
          label: 'Check for Updates',
          click: () => {
            autoUpdater.checkForUpdatesAndNotify();
          }
        },
        {
          label: 'View Documentation',
          click: () => {
            shell.openExternal('https://docs.trustformers.dev');
          }
        },
        {
          label: 'Report Issue',
          click: () => {
            shell.openExternal('https://github.com/trustformers/trustformers-js/issues');
          }
        }
      ]
    }
  ];

  // macOS specific menu adjustments
  if (process.platform === 'darwin') {
    template.unshift({
      label: app.getName(),
      submenu: [
        { role: 'about' },
        { type: 'separator' },
        { role: 'services' },
        { type: 'separator' },
        { role: 'hide' },
        { role: 'hideothers' },
        { role: 'unhide' },
        { type: 'separator' },
        { role: 'quit' }
      ]
    });

    // Window menu
    template[4].submenu = [
      { role: 'close' },
      { role: 'minimize' },
      { role: 'zoom' },
      { type: 'separator' },
      { role: 'front' }
    ];
  }

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
}

// IPC handlers
ipcMain.handle('store:get', (event, key) => {
  return store.get(key);
});

ipcMain.handle('store:set', (event, key, value) => {
  store.set(key, value);
});

ipcMain.handle('store:delete', (event, key) => {
  store.delete(key);
});

ipcMain.handle('app:getVersion', () => {
  return app.getVersion();
});

ipcMain.handle('app:getPlatform', () => {
  return process.platform;
});

ipcMain.handle('dialog:showOpenDialog', async (event, options) => {
  const result = await dialog.showOpenDialog(mainWindow, options);
  return result;
});

ipcMain.handle('dialog:showSaveDialog', async (event, options) => {
  const result = await dialog.showSaveDialog(mainWindow, options);
  return result;
});

ipcMain.handle('dialog:showMessageBox', async (event, options) => {
  const result = await dialog.showMessageBox(mainWindow, options);
  return result;
});

ipcMain.handle('shell:openExternal', (event, url) => {
  shell.openExternal(url);
});

ipcMain.handle('theme:getSystem', () => {
  return nativeTheme.shouldUseDarkColors ? 'dark' : 'light';
});

// App event handlers
app.whenReady().then(() => {
  createSplashWindow();
  
  // Create main window after a short delay
  setTimeout(() => {
    createWindow();
    createMenu();
  }, 1500);

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });

  // Auto-updater
  if (!isDev) {
    autoUpdater.checkForUpdatesAndNotify();
  }
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// Security: Prevent new window creation
app.on('web-contents-created', (event, contents) => {
  contents.on('new-window', (event, navigationUrl) => {
    event.preventDefault();
    shell.openExternal(navigationUrl);
  });
});

// Auto-updater events
autoUpdater.on('checking-for-update', () => {
  console.log('Checking for update...');
});

autoUpdater.on('update-available', (info) => {
  console.log('Update available.');
  
  if (mainWindow) {
    dialog.showMessageBox(mainWindow, {
      type: 'info',
      title: 'Update Available',
      message: 'A new version is available. It will be downloaded in the background.',
      detail: `Version ${info.version} is now available.`
    });
  }
});

autoUpdater.on('update-not-available', (info) => {
  console.log('Update not available.');
});

autoUpdater.on('error', (err) => {
  console.log('Error in auto-updater. ' + err);
});

autoUpdater.on('download-progress', (progressObj) => {
  let log_message = "Download speed: " + progressObj.bytesPerSecond;
  log_message = log_message + ' - Downloaded ' + progressObj.percent + '%';
  log_message = log_message + ' (' + progressObj.transferred + "/" + progressObj.total + ')';
  console.log(log_message);
});

autoUpdater.on('update-downloaded', (info) => {
  console.log('Update downloaded');
  
  if (mainWindow) {
    dialog.showMessageBox(mainWindow, {
      type: 'info',
      title: 'Update Ready',
      message: 'Update downloaded. The application will restart to apply the update.',
      buttons: ['Restart Now', 'Later'],
      defaultId: 0
    }).then(result => {
      if (result.response === 0) {
        autoUpdater.quitAndInstall();
      }
    });
  }
});