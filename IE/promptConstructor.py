import os
from jinja2 import Environment, FileSystemLoader, meta

class PromptConstructor:

    def __init__(self, llmAnnotator):
        self.demos = llmAnnotator.demos
        self.config = llmAnnotator.config
        self.inFileJSON = llmAnnotator.inFileJSON
        self.query = self.inFileJSON.get("text", None)
        self.templ = self.config.templ
    
    def generate_prompt(self):

        try:
            if not self.config.ie_prompt_set or not os.path.isdir(self.config.ie_prompt_set):
                raise ValueError(f"Invalid template directory: {self.config.ie_prompt_set}")
            
            env = Environment(loader=FileSystemLoader(self.config.ie_prompt_set))
            DymTemplate = self.templ
            template_source = env.loader.get_source(env, DymTemplate)[0]
            parsed_content = env.parse(template_source)
            variables = meta.find_undeclared_variables(parsed_content)
            template = env.get_template(DymTemplate)
            
            if variables: 
                if self.demos is not None:
                    Uprompt = template.render(demos=self.demos, query=self.query)
                else:
                    Uprompt = template.render(query=self.query)
            else:
                Uprompt = template.render()

            prompt = [{"role": "user", "content": Uprompt}]
            return prompt
        
        except Exception as e:
            raise RuntimeError(f"Error generating prompt: {e}")