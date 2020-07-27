from Model.GCNN import GCNN
from Model.interpret_v2 import Interpreter
from Model.pymol_attributions import StructureAttribution
from Logger.logger import JSONLogger
import pytorch_lightning as pl
import torch
import os
import json

def main():
	with open('Configs/base_config.json') as base_config:

		my_config = json.load(base_config)
		my_config['input_csv'] = 'Test/input.csv'
		my_config['error_csv'] = 'Test/error.csv'
		my_config['tensors']   = 'Test/Tensors/'
		my_config['pdb']       = 'Test/PDB/'
		my_config['output_name'] = 'Test/Output/'
		my_config['dataset']   = 'MyTestLogger'

		my_config['epochs'] = 2
		# my_config['workers'] = 0

		model   = GCNN(my_config)
		logger  = JSONLogger(
			path="Logs/",
			name=my_config['dataset']
			)
		trainer = pl.Trainer(
			logger=logger, 
			max_epochs=my_config['epochs'],
			checkpoint_callback=False)


		trainer.fit(model)
		trainer.test(model)

	save_path = my_config['output_name']
	torch.save(model.state_dict(), os.path.join(save_path, my_config['dataset'] + '.pt'))
	
	metrics = json.load(open(logger.name + '.json', 'r'))
	
	interpreter = Interpreter(
		model = model,
		loss_fn = torch.nn.CrossEntropyLoss(),
		protein_type = my_config['dataset']
	)

	losses, attributions = interpreter.interpret_test(
		dataset = model.train_dataloader()
	)

	pdbs, output_path = interpreter.generate_attributions(
		losses,
		attributions,
		output_path=os.path.join(save_path, my_config['dataset'] + '.npz')
	)

	print(pdbs, output_path)

	pymol_attr = StructureAttribution(
		attribution_path=output_path,
		data_path=my_config['pdb'],
		output_path=os.path.join(save_path, my_config['dataset'])
	)


	pdb_id = pdbs[0]['pdb_name'] + '_' +pdbs[0]['pdb_chain']
	pdb_id = pdb_id.lower()
	pymol_scene = pymol_attr.structure_attribution(pdb_id)
	
	print(pymol_scene)


if __name__ == '__main__':
	main()
