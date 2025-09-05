export const profile = {
	fullName: 'Archit Agrawal',
	title: '',
	institute: '',
	author_name: '', // Author name to be highlighted in the papers section
	research_areas: [
		// { title: 'Physics', description: 'Brief description of the research interest', field: 'physics' },
	],
}

// Set equal to an empty string to hide the icon that you don't want to display
export const social = {
	email: 'archit.agrawal@nyu.edu',
	linkedin: 'https://www.linkedin.com/in/archit-agrawal-282086145/',
	x: 'https://x.com/architagrawal31',
	github: 'https://github.com/architag',
	gitlab: '',
	scholar: '',
	inspire: '',
	arxiv: '',
	orcid: '',
}

export const template = {
	website_url: 'https://localhost:4321', // Astro needs to know your site’s deployed URL to generate a sitemap. It must start with http:// or https://
	menu_left: false,
	transitions: true,
	lightTheme: 'winter', // Select one of the Daisy UI Themes or create your own
	darkTheme: 'dim', // Select one of the Daisy UI Themes or create your own
	excerptLength: 200,
	postPerPage: 5,
    base: '/' // Repository name starting with /
}

export const seo = {
	default_title: 'Astro Academia',
	default_description: 'Astro Academia is a template for academic websites.',
	default_image: '/images/astro-academia.png',
}
