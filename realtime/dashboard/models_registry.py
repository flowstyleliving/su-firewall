#!/usr/bin/env python3
from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional


def find_project_root(start: str) -> str:
	cur = os.path.abspath(start)
	for _ in range(6):
		candidate = os.path.join(cur, 'config', 'models.json')
		if os.path.isfile(candidate):
			return cur
		parent = os.path.dirname(cur)
		if parent == cur:
			break
		cur = parent
	# Fallback to two levels up from this file
	return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


@dataclass
class FailureLawOverride:
	lambda_: float
	tau: float


@dataclass
class ModelSpec:
	id: str
	display_name: str
	hf_repo: str
	context_len: int
	quant: str
	notes: str
	failure_law: Optional[FailureLawOverride] = None


@dataclass
class ModelsConfig:
	default_model_id: str
	models: List[ModelSpec]


class ModelsRegistry:
	def __init__(self, default_id: str, by_id: Dict[str, ModelSpec]):
		self.default_id = default_id
		self.by_id = by_id

	@classmethod
	def load_from_file(cls, path: Optional[str] = None) -> ModelsRegistry:
		if path and os.path.isfile(path):
			proj_root = os.path.dirname(os.path.dirname(path))
			cfg_path = path
		else:
			proj_root = find_project_root(os.path.dirname(__file__))
			cfg_path = os.path.join(proj_root, 'config', 'models.json')
		with open(cfg_path, 'r', encoding='utf-8') as f:
			data = json.load(f)
		default_id = data.get('default_model_id') or ''
		by_id: Dict[str, ModelSpec] = {}
		for m in data.get('models', []):
			fl = None
			if m.get('failure_law'):
				fl = FailureLawOverride(lambda_=m['failure_law'].get('lambda', 5.0), tau=m['failure_law'].get('tau', 1.0))
			spec = ModelSpec(
				id=m.get('id',''),
				display_name=m.get('display_name',''),
				hf_repo=m.get('hf_repo',''),
				context_len=int(m.get('context_len', 0)),
				quant=m.get('quant',''),
				notes=m.get('notes',''),
				failure_law=fl,
			)
			by_id[spec.id] = spec
		return cls(default_id, by_id)

	def get(self, id_: str) -> Optional[ModelSpec]:
		return self.by_id.get(id_)

	def list(self) -> List[ModelSpec]:
		return list(self.by_id.values()) 