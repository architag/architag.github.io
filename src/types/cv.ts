export interface Experience {
  company: string;
  time: string;
  title: string;
  location?: string;
  description?: string;
}

export interface Education {
  school: string;
  time: string;
  degree: string;
  location?: string;
  description?: string;
}

export interface Skill {
  title: string;
  description: string;
}

export interface Project {
  title: string;
  time: string;
  link?: string;
  abstract?: string;
  details?: ProjectDetail[];
}

export interface ProjectDetail {
  heading: string;
  content: ProjectDetailContent[];
}

export interface ProjectDetailContent {
  itemHeading: string;
  itemContent: string;
}

export function isExperience(element: Experience | Education): element is Experience {
  return 'title' in element && 'company' in element;
}

export function isEducation(element: Education | Experience): element is Education {
  return 'school' in element && 'degree' in element;
}

export function isSkill(element: Skill | Project): element is Skill {
  return 'description' in element;
}

export function isProject(element: Skill | Project): element is Project {
  return 'link' in element;
}
