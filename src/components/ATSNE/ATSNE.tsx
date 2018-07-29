import * as React from 'react';

import { DataSet } from '../../TSNE/data';
import { parseTensorsFromFloat32Array } from '../../TSNE/data-provider';

import './ATSNE.css';

const DATA_FILE = '/data/word2vec_200d.bytes';


/*============ CLASS DEFINITION ============*/

class ATSNE extends React.Component {

  public state = {}

  public componentDidMount () {
    
    const fileReader = new FileReader();

    fileReader.onload = (evt) => {
      let data: Float32Array;
      
      try {
        data = new Float32Array(fileReader.result);
      } catch (e) {
        console.log(e, 'parsing tensor bytes');
        return;
      }

      const dim = 200;
      parseTensorsFromFloat32Array(data, dim).then(dataPoints => {
        const dataSet = new DataSet(dataPoints);
        console.log('dataSet:', dataSet);
      });
    };

    fetch(DATA_FILE)
      .then(res => res.blob())
      .then(blob => {
        fileReader.readAsArrayBuffer(blob);
      });
  };

  public render() {
    return (
        <div>atsne</div>
    );
  }
}


export default ATSNE;