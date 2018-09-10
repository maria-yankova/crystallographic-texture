def read_ctf_header(file_path, no_lines=13):
    """
    Reads the header in a .ctf file.

    Parameters
    ----------
    file : string
        Full file path + name.
    no_lines : int
        Number of lines of header up to and including line for number of phases. 

    Returns
    -------
    header : dict 

    """
   
    n_lines = 13
    with open(file_path) as myfile:
        head = [next(myfile).rstrip('\n') for x in range(n_lines)]

        if head[0] == 'Channel Text File':
            header = {
                'project': head[1].split('\t')[1],
                'authour': head[2].split('\t')[1],
                'JobMode': head[3].split('\t')[1],
                'XCells': int(head[4].split('\t')[1]),
                'YCells': int(head[5].split('\t')[1]),
                'XStep': float(head[6].split('\t')[1]),
                'YStep': float(head[7].split('\t')[1]),
                'AcqE1': float(head[8].split('\t')[1]),
                'AcqE2': float(head[9].split('\t')[1]),
                'AcqE3': float(head[10].split('\t')[1]),
                'data_spec': head[11],
                'no_phases': int(head[12].split('\t')[1]),
                'phases': [],
            }

            phases = [next(myfile).rstrip('\n')
                      for x in range(header['no_phases'])]

            for p_i, p in enumerate(phases):
                latt_params = [float(x) for x in p.split('\t')[0].split(';')
                               + p.split('\t')[1].split(';')]
                name = p.split('\t')[2]

                header['phases'].append({
                    'name': name,
                    'lattice_params': latt_params
                })

        else:
            raise ValueError(
                'Please input the correct file type. File type(s) supported: .ctf.')

        return header
