import * as React from 'react';

import { MuiThemeProvider } from '@material-ui/core/styles';
import { createMuiTheme } from '@material-ui/core/styles';

import Paper from '@material-ui/core/Paper';

import './App.css';

const theme = createMuiTheme({
  palette: {
      type: 'light',
      primary: { main: '#00ff90' },
      secondary: { main: '#042c4b' }
  }
});


/*============ CLASS DEFINITION ============*/

class App extends React.Component {

  public state = {}

  public render() {
    return (

      <MuiThemeProvider theme={theme}>

        <div className='header-container'>
          <header className='container-fluid'>

            <div className='row'>

              <div className='col col-sm-12 text-left'>
                <h1>ATSNE - Anchored TSNE</h1>
                <h2>Preserving spatial relationships via predefined anchor points</h2>
              </div>

            </div>
          </header>
        </div>

        <div className='container-fluid'>
          <div className='row'>


            <div className='col col-sm-12'>
              <Paper>
                <h3>Stuff</h3>
              </Paper>
            </div>

          </div>
        </div>

      </MuiThemeProvider>
    );
  }
}

export default App;