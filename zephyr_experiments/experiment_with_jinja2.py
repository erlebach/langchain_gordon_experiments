from pprint import pprint
import generic_utils as u

rendered_texts = u.apply_templates("scenario_hawking_smolin.yaml")
pprint(rendered_texts)
